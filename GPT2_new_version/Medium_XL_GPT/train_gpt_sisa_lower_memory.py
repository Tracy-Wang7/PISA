"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from model import GPTConfig, GPT
from torch.utils.tensorboard import SummaryWriter
from adam_mini import Adam_mini
from dataclasses import dataclass
#import ipdb

import logger
import io_utils
import uuid
import glob
import time
from collections import defaultdict


# ipdb.set_trace()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
resume_dir = None
eval_interval = 1000
ckpt_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch'
load_iter = 0
# data
dataset = 'fineweb10B' 
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
#optimizer
learning_rate = 0.02 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8
warmup_iters : int = 0
warmdown_iters : int = 1450 
num_iterations : int = 5100
val_tokens : int = 10485760
val_loss_every : int = 125
embed_learning_rate : float = 0.00036
muon_learning_rate : float = 0.002
warmup_iters : int = 0
warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
weight_decay : float = 0
input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
#warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
seed = 1337
comment = 'none'
algorithm = 'muon'
#algorithm = 'adamw'
flash_attn = True
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

print('current dtype', dtype)


save_dir = 'log_gpt2/'+comment


# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Muon optimizer
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
    '''assert len(G.shape) == 2
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
        X = X.T'''
    if len(G.shape) == 2:
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
    else:
        X = G.bfloat16()
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class ADMMParamScheduler:
    def __init__(self, optimizer, ref_optimizer, args, embed_learning_rate, use_residual_adapt=False,
                 mu=10.0, tau=2.0):
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
        self.embed_learning_rate = embed_learning_rate
        self.args = args
        self.sigma0 = args.sigma_lr
        self.rho0 = args.rho_lr
        self.beta_rmsprop = args.beta_rmsprop
        self.mu = mu
        self.tau = tau
        self.rho = args.rho_lr
        self.sigma = args.sigma_lr
        self.step_num = 1

        self.prev_global_weights = None  # store previous global weights if using residual adaptation

    def step(self, global_weights=None, block_weights=None):
        """Call at every iteration to update sigma and rho."""
        t = self.step_num
        lr_current = self.ref_optimizer.param_groups[0]['lr']
        sigma_scaled = (self.embed_learning_rate / lr_current) * self.sigma0
        rho_scaled = (self.embed_learning_rate / lr_current) * self.rho0
        pg = self.optimizer.param_groups[0]

        #self.sigma = sigma_scaled
        #self.rho = rho_scaled

        #if t < self.T - self.D:
        #if t < 175:
        '''if t < 250: #best now
            self.sigma = pg['sigma_lr'] * self.args.gamma
            self.rho = pg['rho_lr'] * self.args.gamma
            if t % 40 == 0:
                print('sigma is:', self.sigma)
                print('rho is:', self.rho)

        

        #elif 250 < t < self.T - self.D:
        elif 375 < t < 626: #step 625 will be better, but later on will become worse
        #elif 375 < t < 501:
            #self.sigma = 175
            #self.rho = 175000
            #self.sigma = 125
            #self.rho = 125000
            self.sigma = pg['sigma_lr'] / 0.99 # 0.99
            self.rho = pg['rho_lr'] / 0.99 # 0.99
            #self.sigma = pg['sigma_lr']
            #self.rho = pg['rho_lr']
            #self.beta_rmsprop = 0.999 # this set of hyperparameters will drop quickly at the beginning, sigma and rho first decrease and then increase
            #self.args.beta1 = 0.75
        
        #elif 250 <=t <= 375 or 500 < t < self.T - self.D:
        #elif 500 < t < self.T - self.D:       
        #elif 625 < t < self.T - self.D:
        #if 500 < t < self.T - self.D:
            #self.sigma = pg['sigma_lr']
            #self.rho = pg['rho_lr']
            #self.beta_rmsprop = 0.999

        


        else:
            self.sigma = sigma_scaled
            self.rho = rho_scaled'''

        
        


        #if 500 < t <875:
        '''if 500 < t <875:
            self.sigma = pg['sigma_lr'] / self.args.gamma
            #self.rho = pg['rho_lr'] / self.args.gamma ### only changes rho_lr later?

        #elif 875 < t < 1000:
        elif 875 < t < self.T - self.D:
            #self.sigma = 450
            self.sigma = 120
            self.beta_rmsprop = 0.999'''

        #elif 1000 < t < self.T - self.D:
            #self.sigma = 200

        '''if 0 < t <= 2250:
            self.sigma = pg['sigma_lr'] / 0.999

        elif 2250 < t < self.T - self.D:
            self.sigma = pg['sigma_lr'] * 0.999'''
        
        '''if 0 < t < self.T - self.D:
            self.sigma = pg['sigma_lr'] / 0.9995

        elif t > self.T - self.D:
            self.sigma = sigma_scaled
            self.rho = rho_scaled
            #self.beta_rmsprop = 0.999
        else:
            #self.sigma = sigma_scaled
            #self.rho = rho_scaled
            self.sigma = pg['sigma_lr']
            self.rho = pg['rho_lr']'''


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
        self.device = torch.cuda.current_device()
        self._step = 1
        self.max_steps = 5100
        self.epsilon_vv = 1e-15
        self.local_batch_size = None  # Will be set during training

        # Maintain original parameter shapes
        self.params = self.param_groups[0]['params']
        #self.W_b = [p.detach().clone() for p in self.params]
        self.P_b = [torch.full_like(p, 1e-10) for p in self.params]
        self.accumulators = [torch.zeros_like(p) for p in self.params]
        self.momentum = [torch.zeros_like(p) for p in self.params]
        #self.prev_W_global = [p.detach().clone() for p in self.params]
        self.total_params = sum(p.numel() for p in self.params)
        #self.updates_flat_w = torch.zeros(self.total_params, device=self.device, dtype=torch.bfloat16)
        #self.updates_flat_p = torch.zeros(self.total_params, device=self.device, dtype=torch.bfloat16)
        self.updates_flat_w = torch.zeros(self.total_params, device=self.device, dtype=torch.float16)

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

        use_bf16 = (self._step > self.max_steps - 2)
        if use_bf16:
            # cast to bfloat16 in a temporary
            self.updates_flat_w = self.updates_flat_w.to(torch.bfloat16)
        else:
            self.updates_flat_w = self.updates_flat_w
        

        # 1. Local parameter updates
        with torch.no_grad():
            sigma_lr = self.param_groups[0]['sigma_lr']
            rho_lr = self.param_groups[0]['rho_lr']
            beta_rmsprop = self.param_groups[0]['beta_rmsprop']
            zeropower_backend = zeropower_backends['newtonschulz5']
            alpha_b = self._get_alpha()

            curr_idx = 0
            
            '''for i, (p, w, pb, acc, v,) in enumerate(zip(
                self.params, self.W_b, self.P_b, self.accumulators, self.momentum
            )):'''
            for i, (p, pb, acc, v,) in enumerate(zip(
                self.params, self.P_b, self.accumulators, self.momentum
            )):
                # layerwise adaptative rho, sigma
                sigma_lr_i = sigma_lr / scales[i] if scales[i] !=0 else sigma_lr
                rho_lr_i = rho_lr   / scales[i] if scales[i] !=0 else rho_lr
                #sigma_lr_i = sigma_lr
                #rho_lr_i = rho_lr
                # Update accumulators
                
                g = p.grad
                if g is None:
                    continue
                v.mul_(self.args.beta1).add_(g) # check whether momentum can improve performance
                '''corrected_v = v / (1 - self.args.beta1**self._step)
                current_grad_v = corrected_v + pb'''               
                #g = p.grad
                g = g.add(v, alpha=self.args.beta1)
                g = zeropower_backend(g, steps=5)
                #g *= max(1, g.size(0)/g.size(1))**0.5
                if len(g.shape) == 2:
                    g *= max(1, g.size(0)/g.size(1))**0.5
                #current_grad = g + pb
                vv = ((pb + g) == 0)
                current_grad = g + pb + self.args.mu * p.data # mu = 1e-1
                '''g = p.grad + pb
                g = zeropower_backend(g, steps=5)
                g *= max(1, g.size(0)/g.size(1))**0.5
                current_grad = g'''

                #current_grad_denom = p.grad + pb + self.args.mu * p.data# p.grad is in the level of 1e-7

                acc.mul_(beta_rmsprop).addcmul_(current_grad, current_grad, value=(1 - beta_rmsprop)) # this current grad should be corrected as the pb + p.grad(which without any modification)
                
                # Compute parameter updates
                #corrected_acc = acc / (1 - beta_rmsprop**self._step)
                corrected_acc = acc
                #correct_acc_top = acc
                #correct_acc_top = 1

                denom = sigma_lr_i + rho_lr_i * (torch.sqrt(corrected_acc) + self.args.epsilon)

                #denom = torch.sqrt(corrected_acc) + self.args.epsilon

                #w.copy_(prev_W_global[i] - (current_grad + self.epsilon_vv**self._step * vv) / denom)
                w = prev_W_global[i] - (current_grad + self.epsilon_vv**self._step * vv) / denom
                pb.copy_((pb.add_(sigma_lr_i * (w - prev_W_global[i]))).mul_(alpha_b)) ### p_scaled = p * alpha
                #w.mul_(alpha_b)
                w *= alpha_b
                #w.copy_(prev_W_global[i] - current_grad / denom)
                #w.copy_(prev_W_global[i] - current_grad / denom).mul_(alpha_b)
                #pb.add_(sigma_lr_i * (w - prev_W_global[i]))

                wpi = sigma_lr*w + pb ### sum first sigma * w + pi first, to reduce memory overhead
                self.updates_flat_w[curr_idx:curr_idx+p.numel()] = wpi.flatten()
                #self.updates_flat_w[curr_idx:curr_idx+p.numel()] = w.flatten()
                #self.updates_flat_p[curr_idx:curr_idx+p.numel()] = pb.flatten()
                curr_idx += p.numel()


                

        # 2. Global aggregation with proper alpha weighting
        dist.all_reduce(self.updates_flat_w, op=dist.ReduceOp.SUM)
        #dist.all_reduce(self.updates_flat_p, op=dist.ReduceOp.SUM)
        
        curr_idx = 0


        # Use all_reduce for direct weighted summation
        #for i, (para, w_global) in enumerate(zip(self.params, self.prev_W_global)):
        for i, para in enumerate(self.params):
            #w_scaled = self.updates_flat_w[curr_idx:curr_idx+para.numel()].view_as(para.data).type_as(para.data)
            #p_scaled = self.updates_flat_p[curr_idx:curr_idx+para.numel()].view_as(para.data).type_as(para.data)
            wpi = self.updates_flat_w[curr_idx:curr_idx+para.numel()].view_as(para.data).type_as(para.data)

            sigma_lr_i = sigma_lr
            #new_W = (sigma_lr_i*w_scaled + p_scaled) / (sigma_lr_i+self.args.l2_lambda)
            new_W = wpi / (sigma_lr_i+self.args.l2_lambda)


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
        lr_t = 0.5*self.adam_opt.param_groups[0]['lr'] # the lr of optimizer2 is half of optimizer1

        # 2) compute the new sigma_lr and rho_lr
        sigma_t = ((1.0 / (self.base_sigma + self.base_rho)) / lr_t) * self.base_sigma
        #rho_t   = (1.0 / lr_t) - sigma_t
        rho_t = ((1.0 / (self.base_sigma + self.base_rho)) / lr_t) * self.base_rho

        # 3) write them back into the DistributedOptimizer’s hyper‐params
        pg = self.dist_opt.param_groups[0]
        pg['sigma_lr'] = sigma_t
        pg['rho_lr']   = rho_t

# -----------------------------------------------------------------------------
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

os.makedirs(save_dir, exist_ok = True)
writer = SummaryWriter(save_dir)


logger_loss_train = logger.Logger('{}/logger_loss_train.txt'.format(save_dir), title='logger_loss_iter')
logger_loss_train.set_names(['iteration', 'trainloss'])
logger_loss_val = logger.Logger('{}/logger_loss_val.txt'.format(save_dir), title='logger_loss_iter')
logger_loss_val.set_names(['iteration', 'valloss'])
logger_loss_time = logger.Logger('{}/logger_loss_time.txt'.format(save_dir), title='logger_time_iter')
logger_loss_time.set_names(['iteration', 'forward backward time', 'clipping time', 'optimizer step time'])

# io_utils.save_code(save_dir)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    #assert gradient_accumulation_steps % ddp_world_size == 0
    #gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 4
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float64': torch.float64}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) 


B, T = batch_size, block_size
# calculate the number of steps to take in the val loop.
assert val_tokens % (B * T * ddp_world_size) == 0
val_steps = val_tokens // (B * T * ddp_world_size)

# poor man's data loader
data_dir = os.path.join('data', dataset)


#train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
#val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


train_loader = DistributedDataLoader(input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(input_val_bin, B, T, ddp_rank, ddp_world_size)

if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0



#load_iter = int(os.environ.get("LOAD_ITER"))
print('load_iter = ', load_iter, 'loading ..', load_iter)

if load_iter == 0:
    init_from = 'scratch'
else: 
    init_from = 'resume'


# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, flash_attn = flash_attn, device = device) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)


elif init_from == 'resume':

    if resume_dir == None:
        resume_dir = out_dir
    print(f"Resuming training from {resume_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(resume_dir, 'ckpt'+str(load_iter)+'.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    print('loading complete')

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
'''if algorithm == 'adamw':
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
elif algorithm == 'adam_mini':
    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        model_sharding=False,
        dim=n_embd,
        n_heads=n_head
    )
elif algorithm == 'muon':
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95, rank=ddp_rank, world_size=ddp_world_size)
    #optimizer.wv_names = {} # For experiments with relatively small total steps  (like the 8B and 13B experiments here, we only run for 10k steps), we apply a single lr for Value and find it performs a bit better. Please comment this line if your total steps is larger than 10k or 20k or more.
    #raise ValueError("algorithm not supported")

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])'''

@dataclass
class Hyperparameters:
    ### this set of sigma_lr and rho_lr keep the same 
    sigma_lr: float = 8e1 # 8e1 best in last version   #8e2 best in current version
    rho_lr: float = 3e3  #3e3   
    beta1: float = 0.95 #0.75 0.95 for best
    beta_rmsprop: float = 0.9 #0.999 and 0.99 does not differ too much
    gamma: float = 0.99
    l2_lambda: float = 0# 1e-1
    mu: float = 0 # 5e-2
    epsilon: float = 1e-8

args = Hyperparameters()
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    #model = torch.nn.DataParallel(model,  device_ids=[ddp_local_rank])
# helps estimate an arbitrarily accurate loss over either split using many batches
raw_model = model.module if ddp else model # unwrap DDP container if needed

# learning rate decay scheduler (cosine with warmup)
# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=embed_learning_rate, betas=(0.9, 0.95),
                               weight_decay=weight_decay, fused=True)

optimizer2_2 = DistributedOptimizer(
    raw_model.transformer.h.parameters(),
    #raw_model.parameters(),
    args
) # it looks the raw_model.transformer.h.parameters() did not get updated


checkpoint = None # free up memory
    
def get_lr(it):
    assert it <= num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return (it+1) / warmup_iters
    # 2) constant lr for a while
    elif it < num_iterations - warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - it) / warmdown_iters
        return decay_ratio
schedulers_adam  = torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr)
sched_sigma_rho = ADMMParamScheduler(optimizer2_2, optimizer1, args, embed_learning_rate)
# logging

def train():
    global iter_num, x, y
    'the following is for training'
    # training loop
    training_time_ms = 0

    torch.cuda.synchronize()
    t0 = time.time()

    local_iter_num = 0 # number of iterations in the lifetime of this process
    
    running_mfu = -1.0
    #while True:
    train_loader.reset()
    val_loss_record = []
    save_path_val = "/workspace0/ow120/DDAM/Adam-mini/examples/gpt2/experiment_results/sisa_submit/val_loss_record.txt"
    train_time_record = []
    save_path_time = "/workspace0/ow120/DDAM/Adam-mini/examples/gpt2/experiment_results/sisa_submit/train_time_record.txt"
    for step in range(num_iterations + 1):
        # determine and set the learning rate for this iteration
        last_step = (step == num_iterations)
        lr = get_lr(iter_num) if decay_lr else learning_rate
        #lr = 0.02
        '''for idd, param_group in enumerate(optimizer.param_groups):
            #param_group['lr'] = lr
            if idd == 0:
                lr = param_group['lr']
                print('The current lr is:', lr)'''
                
            
        

        # evaluate the loss on train/val sets and write checkpoints
        #if iter_num % eval_interval == 0 and master_process:
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        if (last_step or (val_loss_every > 0 and step % val_loss_every == 0)):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps

            if master_process:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f'step:{step}/{num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                val_loss_record.append(val_loss)
                train_time_record.append(training_time_ms)
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()


        if last_step:
            break

        model.train()
        sched_sigma_rho.step()

        with model.no_sync():
            for micro_step in range(1, gradient_accumulation_steps+1):
                optimizer2_2.set_local_batch_size(x.size(0))
                # forward pass
                with ctx:
                    _, loss = model(x, y)
                    train_loss = loss.detach()
                # advance the dataset for the next batch
                x, y = train_loader.next_batch()
                if micro_step < gradient_accumulation_steps:
                    with model.no_sync(): # there's no need to sync gradients every accumulation step
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

            for p in model.parameters():
                    p.grad /= gradient_accumulation_steps

            optimizer2_2.step()


        #synchronize gradients
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            


        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)



        scaler.step(optimizer1)
        scaler.update()
        schedulers_adam.step()
        
        model.zero_grad(set_to_none=True)
                

        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{step+1}/{num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if master_process:
        np.savetxt(save_path_val, np.array([t.cpu().item() for t in val_loss_record]), fmt='%.6f')
        np.savetxt(save_path_time, np.array(train_time_record), fmt='%.6f')
    if ddp:
        destroy_process_group()


train()