from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.policy import A2CPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.pg import TDistFnDiscrOrCont
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor as DiscreteActor
from tianshou.utils.net.discrete import Critic as DiscreteCritic
import sys

class Logger(object):
    def __init__(self, fileN="record.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()                 # flush the file after each write
    def flush(self):
        self.log.flush()
        
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/HalfCheetahV3_2.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/AntV3_4.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/HumanoidV3_2.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/InvertedDoublePendulumV2_2.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/Walker2dV2_2.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/Hopper_1.txt")
sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/Swimmer_2.txt")
@dataclass(kw_only=True)
class PPOTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    clip_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats
    gradient_steps: int = 0

    @classmethod
    def from_sequences(
        cls,
        *,
        losses: Sequence[float],
        clip_losses: Sequence[float],
        vf_losses: Sequence[float],
        ent_losses: Sequence[float],
        gradient_steps: int = 0,
    ) -> Self:
        return cls(
            loss=SequenceSummaryStats.from_sequence(losses),
            clip_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
        )


TPPOTrainingStats = TypeVar("TPPOTrainingStats", bound=PPOTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class PPOPolicy_SPiAM(A2CPolicy[TPPOTrainingStats], Generic[TPPOTrainingStats]):  # type: ignore[type-var]
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param actor: the actor network following the rules:
        If `self.action_type == "discrete"`: (`s` ->`action_values_BA`).
        If `self.action_type == "continuous"`: (`s` -> `dist_input_BD`).
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
    :param eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper.
    :param dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound. Set to None
        to disable dual-clip PPO.
    :param value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
    :param advantage_normalization: whether to do per mini-batch advantage
        normalization.
    :param recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
    :param vf_coef: weight for value loss.
    :param ent_coef: weight for entropy loss.
    :param max_grad_norm: clipping gradients in back propagation.
    :param gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
    :param max_batchsize: the maximum size of the batch when computing GAE.
    :param discount_factor: in [0, 1].
    :param reward_normalization: normalize estimated values to have std close to 1.
    :param deterministic_eval: if True, use deterministic evaluation.
    :param observation_space: the space of the observation.
    :param action_scaling: if True, scale the action from [-1, 1] to the range of
        action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module | ActorProb | DiscreteActor,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        optim: torch.optim.Optimizer,
        dist_fn: TDistFnDiscrOrCont,
        action_space: gym.Space,
        eps_clip: float = 0.2,
        dual_clip: float | None = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        num_gpu: int = 4,
        beta_rmsprop: float = 0.999,
        lr: float = 3e-4,
        l2_lambda: float = 0.0000,
        sigma_lr: float = 0.005,
        eps: float = 1e-8,

    ) -> None:
        assert (
            dual_clip is None or dual_clip > 1.0
        ), f"Dual-clip PPO parameter should greater than 1.0 but got {dual_clip}"

        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,)
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        self.norm_adv = advantage_normalization
        self.recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic
        self.num_gpu=num_gpu
        self.beta_rmsprop=beta_rmsprop
        self.lr=lr
        self.sigma_lr=sigma_lr
        self.eps=eps
        self.l2_lambda=l2_lambda

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        if self.recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        batch: LogpOldProtocol
        return batch

    def average_parameters(self, num_train_env, list_vars, list_alpha):
        sum_vars = [torch.zeros_like(var) for var in list_vars[0]]
        for i in range(num_train_env):
            W_n = list_vars[i]
            alpha = list_alpha[i]
            sum_vars = [sum_ + alpha * update for sum_, update in zip(sum_vars, W_n)]
        return sum_vars

    def generate_W_global(self, num_batches, W_n_list, P_n_list, tau_lr, alpha, l2_lambda_):
        W_n_avg = self.average_parameters(num_batches, W_n_list, alpha)
        P_n_avg = self.average_parameters(num_batches, P_n_list, alpha)
        for i in range(len(W_n_avg)):
            #W_n_avg[i] = W_n_avg[i] + P_n_avg[i] / tau_lr
            W_n_avg[i] = ((np.float64(tau_lr)*W_n_avg[i].double())/ (np.float64(tau_lr) + np.float64(l2_lambda_))).float() + P_n_avg[i]/(tau_lr + l2_lambda_)
            W_n_avg[i].detach()

        # del P_n_avg
        # gc.collect()
        return W_n_avg

    def zero_grad(self, params):
        """
        Zeroes out gradients for the given parameters.
        """
        for param in params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def compute_learning_rate(self, current_epoch, step_per_epoch, step_per_collect, total_epochs):
        max_update_num = np.ceil(step_per_epoch / step_per_collect) * total_epochs
        return max(0.0, 1 - current_epoch / max_update_num)

    # TODO: why does mypy complain?
    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPPOTrainingStats:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        updated_iteration = 1.0
        #split_batch_size = batch_size or -1
        split_batch_size = batch_size *4 or -1
        split_sub_batch_size = split_batch_size//self.num_gpu
        #print('The split_batch_size is:', split_sub_batch_size)

        ##define trainable parameters for SPiAM
        W_n_0 = [param.clone().detach().requires_grad_(True) for param in self._actor_critic.parameters()]
        W_b_initial = [[param.clone() for param in W_n_0] for _ in range(self.num_gpu)]
        P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(self.num_gpu)]
        accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(self.num_gpu)]
        velocities_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(self.num_gpu)]

        alpha_b = [1 / self.num_gpu for _ in range(self.num_gpu)]
        sigma_lr = self.sigma_lr
        W_global = self.generate_W_global(self.num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b, self.l2_lambda)

        self.zero_grad(self._actor_critic.parameters())

        #learning_rate_current = self.optim.param_groups[0]['lr']  ### check whether the learning rate keep changes
        learning_rate_current = self.lr_scheduler.get_last_lr()[0]
        sigma_lr_current = (self.lr / learning_rate_current) * self.sigma_lr
        rho_lr_current = 1 / learning_rate_current - sigma_lr_current

        for step in range(repeat):

            if self.recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)

            clip_loss_avg = 0
            vf_loss_avg = 0
            ent_loss_avg = 0
            loss_avg = 0
            
            
            for _batch in batch.split(split_batch_size, merge_last=True):
                

                for batch_idx, minibatch in enumerate(_batch.split(split_sub_batch_size, merge_last=True)):
                
                    with torch.no_grad():  # Disable gradient tracking
                        for param, w in zip(self._actor_critic.parameters(), W_global):
                            param.copy_(w)
                
                    W_n = W_b_initial[batch_idx]
                    P_n = P_b_initial[batch_idx]
                    accumulators = accumulators_initial[batch_idx]
                    velocities = velocities_initial[batch_idx]
    
                    # calculate loss for actor
                    advantages = minibatch.adv
                    dist = self(minibatch).dist
                    if self.norm_adv:
                        mean, std = advantages.mean(), advantages.std()
                        advantages = (advantages - mean) / (std + self._eps)  # per-batch norm
                    ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                    ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                    surr1 = ratios * advantages
                    surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                    if self.dual_clip:
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(clip1, self.dual_clip * advantages)
                        clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                    else:
                        clip_loss = -torch.min(surr1, surr2).mean()
                    # calculate loss for critic
                    value = self.critic(minibatch.obs).flatten()
                    if self.value_clip:
                        v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                            -self.eps_clip,
                            self.eps_clip,
                        )
                        vf1 = (minibatch.returns - value).pow(2)
                        vf2 = (minibatch.returns - v_clip).pow(2)
                        vf_loss = torch.max(vf1, vf2).mean()
                    else:
                        vf_loss = (minibatch.returns - value).pow(2).mean()
                    # calculate regularization and overall loss
                    ent_loss = dist.entropy().mean()
                    loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                    #self.optim.zero_grad()
    
                    clip_loss_avg += clip_loss.item()
                    vf_loss_avg += vf_loss.item()
                    ent_loss_avg += ent_loss.item()
                    loss_avg += loss.item()
    
                    self.zero_grad(self._actor_critic.parameters())
                    loss.backward()
                    if self.max_grad_norm:  # clip large gradient
                        nn.utils.clip_grad_norm_(
                            self._actor_critic.parameters(),
                            max_norm=self.max_grad_norm,
                        )
                    gradients = [param.grad for param in self._actor_critic.parameters()]
                    #self.optim.step()
                    ##### SPiAM optimization
                    with torch.no_grad():
    
                        for i, (param_wn, param_pn, gradient, param_wg, accumulator, velocity) in enumerate(
                                zip(W_n, P_n, gradients, W_global, accumulators, velocities)):
                            # velocity.mul_(args.beta1).add_((1 - args.beta1) * (gradient + param_pn))
                            accumulator.mul_(self.beta_rmsprop).add_((1 - self.beta_rmsprop) * (gradient + param_pn).pow(2))
                            # accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) * gradient.pow(2))
                            # accumulator.mul_(beta_rmsprop).add_((1 - beta_rmsprop) *  param_pn.pow(2))
    
                            # bias_correction1 = 1 - args.beta1** updated_iteration
                            # corrected_velocity = velocity / (bias_correction1)
                            bias_correction2 = 1 - self.beta_rmsprop ** updated_iteration
                            corrected_accumulator = accumulator / (bias_correction2)
                            delta = param_wg - (gradient + param_pn) / (
                                        sigma_lr_current + rho_lr_current * (torch.sqrt(corrected_accumulator) + self.eps))
                            # delta = param_wg -  args.lr*(gradient + param_pn)/(torch.sqrt(corrected_accumulator) + args.eps)
    
                            param_wn.copy_(delta.detach())
                            param_pn.add_(sigma_lr_current * (param_wn - param_wg))
    
                    # zero_grad(model.parameters())
                    del loss

                updated_iteration += 1
                with torch.no_grad():
                    W_global = self.generate_W_global(self.num_gpu, W_b_initial, P_b_initial, sigma_lr_current, alpha_b, self.l2_lambda)
                    for param, w in zip(self._actor_critic.parameters(), W_global):
                        param.copy_(w)





                self.optim.step() ### This is only defined for lr_scheduler
                clip_losses.append(clip_loss_avg/self.num_gpu)
                vf_losses.append(vf_loss_avg/self.num_gpu)
                ent_losses.append(ent_loss_avg/self.num_gpu)
                losses.append(loss_avg/self.num_gpu)
    
                gradient_steps+=1

        return PPOTrainingStats.from_sequences(  # type: ignore[return-value]
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=vf_losses,
            ent_losses=ent_losses,
            gradient_steps=gradient_steps,
        )