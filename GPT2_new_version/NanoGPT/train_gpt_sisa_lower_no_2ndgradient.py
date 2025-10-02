import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon optimizer
def simple_adam_step(params, grads, exp_avg, exp_avg_sq, step,
                     lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    """
    Performs a single Adam update on the given parameters.

    Arguments:
        params     (list[Tensor]): parameters to update (i.e. model.parameters())  
        grads      (list[Tensor]): corresponding gradients (p.grad for p in params)  
        exp_avg    (list[Tensor]): first moment buffers (same shapes as params)  
        exp_avg_sq (list[Tensor]): second moment buffers  
        step       (int): current time step (1-indexed)  
        lr         (float): learning rate  
        betas      (tuple[float, float]): (beta1, beta2) decay rates  
        eps        (float): term added to denominator for numerical stability  
    """

    beta1, beta2 = betas
    for p, g, m, v in zip(params, grads, exp_avg, exp_avg_sq):
        if g is None:
            continue

        # 1) Update biased first moment estimate.
        m.mul_(beta1).add_(g, alpha=(1 - beta1))
        # 2) Update biased second raw moment estimate.
        v.mul_(beta2).addcmul_(g, g, value=(1 - beta2))

        # 3) Compute bias-corrected estimates.
        #m_hat = m / (1 - beta1 ** step)
        m_hat = g
        v_hat = v / (1 - beta2 ** step)


        # 4) Update parameters.
        update = m_hat / (v_hat.sqrt() + eps)
        p.data.add_(update, alpha=-lr)


import math
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class ADMMParamScheduler:
    def __init__(self, optimizer, ref_optimizer, args, use_residual_adapt=False, tau=2.0):
        """
        ADMM scheduler to update sigma and rho over training steps.

        Args:
            optimizer: Your ADMM-based optimizer.
            num_iterations: Total number of training iterations.
            sigma0: Initial (max) value of sigma.
            rho0: Initial (max) value of rho.
            warmup_iters: Number of warm-up steps for sigma and rho.
            hold_iters: Number of steps to hold before decay.
            use_residual_adapt: Whether to adapt rho based on residuals.
            mu: Residual adaptation threshold multiplier.
            tau: Residual adaptation update factor.
        """
        self.optimizer = optimizer
        self.ref_optimizer = ref_optimizer
        self.args = args
        self.sigma0 = args.sigma_lr
        self.rho0 = args.rho_lr
        self.beta_rmsprop = args.beta_rmsprop
        self.T = args.num_iterations
        self.W = args.warmup_iters
        self.D = args.warmdown_iters
        self.tau = tau
        self.rho = args.rho_lr
        self.sigma = args.sigma_lr
        self.step_num = 0

        self.prev_global_weights = None  # store previous global weights if using residual adaptation

    def step(self, global_weights=None, block_weights=None):
        """Call at every iteration to update sigma and rho."""
        t = self.step_num
        lr_current = self.ref_optimizer.param_groups[0]['lr']
        '''sigma_scaled = (self.args.embed_learning_rate / lr_current) * self.sigma0
        rho_scaled = (self.args.embed_learning_rate / lr_current) * self.rho0'''
        if t < args.num_iterations - args.warmdown_iters:
            #self.sigma0 /= self.args.gamma
            sigma_scaled = (self.args.embed_learning_rate / lr_current) * self.sigma0
            rho_scaled = (self.args.embed_learning_rate / lr_current) * self.rho0

            
            
        #rho_scaled = self.rho0
        else:
            decay_ratio = (args.num_iterations - t) / args.warmdown_iters
            sigma_scaled = (self.args.embed_learning_rate / lr_current) * self.sigma0
            rho_scaled = (self.args.embed_learning_rate / lr_current) * self.rho0
            sigma_scaled /= decay_ratio 
            #rho_scaled /= decay_ratio # no rho_scaled best currently

        pg = self.optimizer.param_groups[0]


        self.sigma = sigma_scaled
        self.rho = rho_scaled
        pg['sigma_lr'] = self.sigma
        pg['rho_lr']   = self.rho
        pg['beta_rmsprop']   = self.beta_rmsprop


        self.step_num += 1

    def _residual_update(self, global_weights, block_weights):
        """
        Optionally adapt rho based on primal and dual residual norms.
        """
        r_norm = sum((wb - global_weights).norm() ** 2 for wb in block_weights).sqrt()
        if self.prev_global_weights is not None:
            s_norm = self.rho * (global_weights - self.prev_global_weights).norm()
            if r_norm > self.mu * s_norm:
                self.rho *= self.tau
            elif s_norm > self.mu * r_norm:
                self.rho /= self.tau

        self.prev_global_weights = global_weights.detach().clone()
        # Keep sigma in same proportion
        self.sigma = self.sigma0 * (self.rho / self.rho0)


class DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, args):
        super().__init__(params, 
                        dict(sigma_lr=args.sigma_lr,
                            rho_lr=args.rho_lr,
                            beta_rmsprop=args.beta_rmsprop,
                            l2_lambda = args.l2_lambda,
                            epsilon=args.epsilon))
        
        self.num_gpu = torch.cuda.device_count()
        self.args = args
        self.rank = dist.get_rank()
        self.world_size = 4
        self.device = torch.cuda.current_device()
        self._step = 1
        self.max_steps = 5100
        self.epsilon_vv = 1e-15
        self.local_batch_size = None  # Will be set during training

        # Maintain original parameter shapes
        self.params = self.param_groups[0]['params']
        #self.W_b = [p.detach().clone() for p in self.params]
        #self.P_b = [torch.full_like(p, 1e-10) for p in self.params]
        #self.accumulators = [torch.zeros_like(p) for p in self.params]
        #self.momentum = [torch.zeros_like(p) for p in self.params]
        #self.total_params = sum(p.numel() for p in self.params)
        #self.updates_flat_w = torch.zeros(self.total_params, device=self.device, dtype=torch.bfloat16)
        #self.updates_flat_p = torch.zeros(self.total_params, device=self.device, dtype=torch.bfloat16)
        #self.updates_flat_w = torch.zeros(self.total_params, device=self.device, dtype=torch.float16)

    def set_local_batch_size(self, batch_size):
        """Call this before each batch to set the local batch size"""
        self.local_batch_size = batch_size

    def _compute_layer_penalties(self):
        # Compute per-layer scale = ‖W‖ / mean_j ‖W_j‖
        norms = torch.tensor([p.data.norm() for p in self.params], device=self.params[0].device)
        mean_norm = norms.mean()
        scales = norms / (mean_norm + 1e-16)
        scales = torch.clamp(scales, min=0.1, max=10.0)
        return scales  # shape = [num_params]

    def _get_alpha(self):
        """Calculate alpha based on local batch size fraction"""
        # Gather all batch sizes across devices
        batch_sizes = [torch.tensor(0, device=self.device) for _ in range(self.num_gpu)]
        dist.all_gather(batch_sizes, torch.tensor(self.local_batch_size, device=self.device))
        total_batch = sum(b.item() for b in batch_sizes)
        
        # Calculate alpha for this device
        return self.local_batch_size / total_batch

    def step(self, normalized_factor = 1.0, closure=None):
        # Verify batch size was set
        if self.local_batch_size is None:
            raise RuntimeError("Must call set_local_batch_size() before step()")

        scales = self._compute_layer_penalties()
        # Store previous global parameters
        prev_W_global = [w.clone() for w in self.params]

        use_bf16 = (self._step > self.max_steps - 3)
        '''if use_bf16:
            # cast to bfloat16 in a temporary
            self.updates_flat_w = self.updates_flat_w.to(torch.bfloat16)
        else:
            self.updates_flat_w = self.updates_flat_w'''
        

        # 1. Local parameter updates
        with torch.no_grad():
            sigma_lr = self.param_groups[0]['sigma_lr']
            rho_lr = self.param_groups[0]['rho_lr']
            beta_rmsprop = self.param_groups[0]['beta_rmsprop']
            zeropower_backend = zeropower_backends['newtonschulz5']
            alpha_b = self._get_alpha()
            updates_flat_w = torch.zeros(sum(p.numel() for p in self.params), device='cuda', dtype=torch.bfloat16)
            curr_idx = 0

            if use_bf16:
                # cast to bfloat16 in a temporary
                updates_flat_w = updates_flat_w.to(torch.bfloat16)
            else:
                updates_flat_w = updates_flat_w

            '''for i, (p, pb, acc, v,) in enumerate(zip(
                self.params, self.P_b, self.accumulators, self.momentum
            )):'''
            '''for i, (p, pb, acc) in enumerate(zip(
                self.params, self.P_b, self.accumulators
            )):'''
            '''for i, (p, pb) in enumerate(zip(
                self.params, self.P_b
            )):'''
            for i, p in enumerate(self.params):

                # layerwise adaptative rho, sigma
                #sigma_lr_i = sigma_lr / scales[i] if scales[i] !=0 else sigma_lr
                #rho_lr_i = rho_lr   / scales[i] if scales[i] !=0 else rho_lr
                sigma_lr_i = sigma_lr
                rho_lr_i = rho_lr
                # Update accumulators
                
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']

                if 'multiplier_buffer' not in state:
                    #state['multiplier_buffer'] = torch.zeros_like(g)
                    state['multiplier_buffer'] = torch.full_like(p.data, 1e-10)
                pb = state['multiplier_buffer']  #### memory will not rise
                buf.mul_(self.args.beta1).add_(g)
                #v.mul_(self.args.beta1).add_(g) # check whether momentum can improve performance
                '''corrected_v = v / (1 - self.args.beta1**self._step)
                current_grad_v = corrected_v + pb'''               
                #g = p.grad
                #g = g.add(v, alpha=self.args.beta1)
                g = g.add(buf, alpha=self.args.beta1)
                g = zeropower_backend(g, steps=5)
                #g *= max(1, g.size(0)/g.size(1))**0.5
                if len(g.shape) == 2:
                    g *= max(1, g.size(0)/g.size(1))**0.5
                #current_grad = g + pb
                '''if 'multiplier_buffer' not in state:
                    #state['multiplier_buffer'] = torch.full_like(p.data, 1e-10)
                    state['multiplier_buffer'] = torch.zeros_like(g)
                pb = state['multiplier_buffer']  #### will memory rise'''

                #pb = acc


                vv = ((pb + g) == 0)
                #vv = False
                #current_grad = g + self.args.mu * p.data
                current_grad = g + pb + self.args.mu * p.data # mu = 1e-1

                '''g = p.grad + pb
                g = zeropower_backend(g, steps=5)
                g *= max(1, g.size(0)/g.size(1))**0.5
                current_grad = g'''
                '''if 'second_moment_buffer' not in state:
                    state['second_moment_buffer'] = torch.zeros_like(g)
                acc = state['second_moment_buffer']'''

                #acc.mul_(beta_rmsprop).addcmul_(current_grad, current_grad, value=(1 - beta_rmsprop)) # this current grad should be corrected as the pb + p.grad(which without any modification)
                

                denom = sigma_lr_i + rho_lr_i*(torch.sqrt(current_grad*current_grad) + self.args.epsilon)
                #denom = sigma_lr_i + rho_lr_i
                '''if self._step < self.args.num_iterations - self.args.warmdown_iters:
                    denom = sigma_lr_i + rho_lr_i
                else:
                    denom = sigma_lr_i + rho_lr_i*(torch.sqrt(current_grad*current_grad) + self.args.epsilon)'''
                    


                #denom = torch.sqrt(corrected_acc) + self.args.epsilon

                w = prev_W_global[i] - (current_grad + self.epsilon_vv**self._step * vv) / denom
                #w = prev_W_global[i] - current_grad / (denom)
                pb.copy_((pb.add_(sigma_lr_i * (w - prev_W_global[i]))).mul_(alpha_b)) ### p_scaled = p * alpha

                w *= alpha_b

                wpi = sigma_lr*w + pb ### sum first sigma * w + pi first, to reduce memory overhead
                #wpi = w

                updates_flat_w[curr_idx:curr_idx+p.numel()] = wpi.flatten()

                curr_idx += p.numel()


                

        # 2. Global aggregation with proper alpha weighting
        dist.all_reduce(updates_flat_w, op=dist.ReduceOp.SUM)
        #dist.all_reduce(self.updates_flat_p, op=dist.ReduceOp.SUM)
        
        curr_idx = 0


        # Use all_reduce for direct weighted summation
        #for i, (para, w_global) in enumerate(zip(self.params, self.prev_W_global)):
        for i, para in enumerate(self.params):
            #w_scaled = self.updates_flat_w[curr_idx:curr_idx+para.numel()].view_as(para.data).type_as(para.data)
            #p_scaled = self.updates_flat_p[curr_idx:curr_idx+para.numel()].view_as(para.data).type_as(para.data)
            wpi = updates_flat_w[curr_idx:curr_idx+para.numel()].view_as(para.data).type_as(para.data)

            sigma_lr_i = sigma_lr
            #new_W = (sigma_lr_i*w_scaled + p_scaled) / (sigma_lr_i+self.args.l2_lambda)
            new_W = wpi / (sigma_lr_i+self.args.l2_lambda)
            #new_W = wpi


            para.data.copy_(new_W)

            curr_idx += para.numel()
        

        self._step += 1

class SigmaRhoScheduler:
    """
    Schedule sigma_lr and rho_lr in a DistributedOptimizer to track
    the 'real' learning rate of a paired AdamW optimizer.
    """
    def __init__(self,
                 dist_opt: DistributedOptimizer,
                 adam_opt: torch.optim.Optimizer,
                 base_sigma: float,
                 base_rho: float):
        self.dist_opt     = dist_opt
        self.adam_opt     = adam_opt
        self.base_sigma   = base_sigma
        self.base_rho     = base_rho

    def step(self):
        # 1) pull the current lr from the AdamW optimizer
        #lr_t = 0.5*self.adam_opt.param_groups[0]['lr'] # the lr of optimizer2 is half of optimizer1

        # 2) compute the new sigma_lr and rho_lr
        #sigma_t = ((1.0 / (self.base_sigma + self.base_rho)) / lr_t) * self.base_sigma
        sigma_t = self.base_sigma/(self.adam_opt.param_groups[0]['lr']/self.embed_learning_rate)
        #rho_t   = (1.0 / lr_t) - sigma_t
        #rho_t = ((1.0 / (self.base_sigma + self.base_rho)) / lr_t) * self.base_rho
        rho_t = self.base_rho/(self.adam_opt.param_groups[0]['lr']/self.embed_learning_rate)

        # 3) write them back into the DistributedOptimizer’s hyper‐params
        pg = self.dist_opt.param_groups[0]
        pg['sigma_lr'] = sigma_t
        pg['rho_lr']   = rho_t
# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    #batch_size : int = 2*64 # batch size, in sequences, across all devices
    #device_batch_size : int = 64 # batch size, in sequences, per device
    batch_size : int = 4*16
    device_batch_size : int = 16
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    embed_learning_rate : float = 0.0036
    muon_learning_rate : float = 0.02
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # 1450 number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    hold_iters: int = 1000
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    #sigma_lr: float = 5e1 
    #rho_lr: float = 5e4 
    ### this set of sigma_lr and rho_lr keep the same 
    sigma_lr: float = 8e1 # 8e1 best in last version   #8e2 best in current version
    rho_lr: float =1e2  #1e2 1.5e2 bad 8e1 also bad
    beta1: float = 0.95 #0.75 0.95 for best
    beta_rmsprop: float = 0.9 #0.999 and 0.99 does not differ too much
    gamma: float = 0.999
    l2_lambda: float = 0# 1e-1
    mu: float = 0 # 5e-2
    epsilon: float = 1e-8

@dataclass
class Hyperparameters_1:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    #batch_size : int = 2*64 # batch size, in sequences, across all devices
    #device_batch_size : int = 64 # batch size, in sequences, per device
    batch_size : int = 4*16
    device_batch_size : int = 16
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    embed_learning_rate : float = 0.0036
    muon_learning_rate : float = 0.02
    warmup_iters : int = 250
    warmdown_iters : int = 1950 # 1450 number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    hold_iters: int = 1000
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    sigma_lr: float = 2.5e-3  # 2.5e-3
    rho_lr: float = 5e4
    beta1: float = 0.5
    beta_rmsprop: float = 0.99
    l2_lambda: float =1e-6
    epsilon: float = 1e-12  

args = Hyperparameters()
args_1 = Hyperparameters_1()
# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
#model = torch.compile(model)
val_loss_record = []
save_path_val = "/workspace0/ow120/DDAM/modded-nanogpt/experiment_results/sisa_submit/val_loss_record_no_2ndgradient.txt"
train_time_record = []
save_path_time = "/workspace0/ow120/DDAM/modded-nanogpt/experiment_results/sisa_submit/train_time_record_no_2ndgradient.txt"
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)

optimizer2_1 = DistributedOptimizer(
    raw_model.lm_head.parameters(),
    args=args_1
)

optimizer2_2 = DistributedOptimizer(
    raw_model.transformer.h.parameters(),
    #raw_model.parameters(),
    args=args
) # it looks the raw_model.transformer.h.parameters() did not get updated
optimizer2 = torch.optim.Adam(raw_model.transformer.h.parameters(), lr=0.5*args.embed_learning_rate, betas=(0.9, 0.95),
                                weight_decay=args.weight_decay, fused=True)

#optimizer2 = torch.optim.RMSprop(raw_model.transformer.h.parameters(), lr=0.5*args.embed_learning_rate, alpha=0.95, weight_decay=args.weight_decay)
#optimizers = [optimizer1, optimizer2]
'''optimizer2 = DistributedOptimizer(
    raw_model.parameters(),
    args=args
)'''
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
#schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
schedulers_adam  = torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr)
schedulers_adam2  = torch.optim.lr_scheduler.LambdaLR(optimizer2, get_lr)
'''sched_sigma_rho = SigmaRhoScheduler(
    dist_opt   = optimizer2_2,
    adam_opt   = optimizer1,
    base_sigma = args.sigma_lr,
    base_rho   = args.rho_lr
)'''
sched_sigma_rho = ADMMParamScheduler(optimizer2_2, optimizer1, args)
# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()


'''exp_avg    = [torch.zeros_like(p) for p in raw_model.transformer.h.parameters()]
exp_avg_sq = [torch.zeros_like(p) for p in raw_model.transformer.h.parameters()]
iteration_step = 0'''

try:

    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                print('The sigma_lr is:', optimizer2_2.param_groups[0]['sigma_lr'])
                val_loss_record.append(val_loss.data)
                train_time_record.append(training_time_ms)

                #with open(logfile, "a") as f:
                    #f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            #log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            #torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        #iteration_step += 1
        sched_sigma_rho.step()
        #param_check = list(raw_model.transformer.h.parameters())
        #print('The admm update parameter is:', param_check[0][0:8])
        #print('#############################################')
        for i in range(1, train_accumulation_steps+1):
            
            # forward pass
            optimizer2_2.set_local_batch_size(x.size(0))
            #optimizer2_1.set_local_batch_size(x.size(0))
            '''with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            if i < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward() # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps'''
            with model.no_sync():
                with ctx:
                    _, loss = model(x, y, return_logits=False)
                    train_loss = loss.detach()
                # advance the dataset for the next batch
                x, y = train_loader.next_batch()

                loss.backward()
                #if step == 0:
                    #normalized_factor = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in raw_model.transformer.h.parameters() if p.grad is not None))

                

                optimizer2_2.step()
                
        '''for p in raw_model.lm_head.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        for p in raw_model.transformer.h.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)'''
        # sync gradient in all devices, since for sisa it keeps no_sync
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                #dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        '''grad = next(p.grad for p in raw_model.lm_head.parameters() if p.requires_grad)
        print(f"Rank {dist.get_rank()}, Grad Norm: {grad.norm()}")
        print('?????????????????????????????????')'''
        optimizer1.step()
        #optimizer2_2.step()
        #optimizer2.step()
        schedulers_adam.step()
        #schedulers_adam2.step()
        #sched_sigma_rho_1.step()
        #sched_sigma_rho.step()
        #print('The adam learning rate is:', 0.5*optimizer1.param_groups[0]['lr'])
        #print('The SISA learning rate is:', optimizer2_2.param_groups[0]['rho_lr'])
        #print('The current sisa lr is:', 1/(optimizer2_2.param_groups[0]['sigma_lr']+optimizer2_2.param_groups[0]['rho_lr']))
        '''for idx, p in enumerate(raw_model.transformer.h.parameters()):
            if idx == 0:
                print('check whether weight is updated:', p[0][0:8])
                print('###########################################')'''
        
        
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

            '''current_sigma_lr = optimizer2_2.param_groups[0]['sigma_lr']
            current_rho_lr = optimizer2_2.param_groups[0]['rho_lr']
            print(f"step:{step+1}/{args.num_iterations}, current lr: {1/(current_sigma_lr+current_rho_lr)}")'''

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        print(val_loss_record)
        #np.savetxt(save_path_val, np.array(val_loss_record), fmt='%.6f')
        np.savetxt(save_path_val, np.array([t.cpu().item() for t in val_loss_record]), fmt='%.6f')
        np.savetxt(save_path_time, np.array(train_time_record), fmt='%.6f')
        #np.savetxt(save_path_time, np.array([t.cpu().item() for t in train_time_record]), fmt='%.6f')


    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()

except KeyboardInterrupt:
    print("\nTraining interrupted. Here's the val results:")
    print(val_loss_record)
    exit(0)  # or sys.exit(0)
