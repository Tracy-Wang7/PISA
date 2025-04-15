from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.policy import PGPolicy
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
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/Walker2dV3_2.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/Hopper_1.txt")
#sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/SwimmerV3_3.txt")
sys.stdout = Logger("/workspace1/ow120/DDAM/RL/tianshou/examples/mujoco/InvertedDoublePendulum_2.txt")
@dataclass(kw_only=True)
class A2CTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats


TA2CTrainingStats = TypeVar("TA2CTrainingStats", bound=A2CTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class A2CPolicy_SPiAM(PGPolicy[TA2CTrainingStats], Generic[TA2CTrainingStats]):  # type: ignore[type-var]
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    :param actor: the actor network following the rules:
        If `self.action_type == "discrete"`: (`s_B` ->`action_values_BA`).
        If `self.action_type == "continuous"`: (`s_B` -> `dist_input_BD`).
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
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
        Only used if the action_space is continuous.
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
        super().__init__(
            actor=actor,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, f"GAE lambda should be in [0, 1] but got: {gae_lambda}"
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.max_batchsize = max_batchsize
        self._actor_critic = ActorCritic(self.actor, self.critic)
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
    ) -> BatchWithAdvantagesProtocol:
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
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
    
    
    def _compute_returns(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
                v_s_.append(self.critic(minibatch.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Empirical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        # TODO: see todo in PGPolicy.process_fn
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    # TODO: mypy complains b/c signature is different from superclass, although
    #  it's compatible. Can this be fixed?
    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TA2CTrainingStats:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        updated_iteration = 1.0
        #split_batch_size = batch_size or -1
        split_batch_size = batch_size *4 or -1
        split_sub_batch_size = split_batch_size//self.num_gpu
        
        
        W_n_0 = [param.clone().detach().requires_grad_(True) for param in self._actor_critic.parameters()]
        W_b_initial = [[param.clone() for param in W_n_0] for _ in range(self.num_gpu)]
        P_b_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(self.num_gpu)]
        accumulators_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(self.num_gpu)]
        velocities_initial = [[torch.zeros_like(param) for param in W_n_0] for _ in range(self.num_gpu)]

        alpha_b = [1 / self.num_gpu for _ in range(self.num_gpu)]
        sigma_lr = self.sigma_lr
        W_global = self.generate_W_global(self.num_gpu, W_b_initial, P_b_initial, sigma_lr, alpha_b, self.l2_lambda)

        self.zero_grad(self._actor_critic.parameters())
        
        
        learning_rate_current = self.lr_scheduler.get_last_lr()[0]
        sigma_lr_current = (self.lr / learning_rate_current) * self.sigma_lr
        rho_lr_current = 1 / learning_rate_current - sigma_lr_current
        
        for _ in range(repeat):
        
            actor_loss_avg = 0
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
                    
                    dist = self(minibatch).dist
                    log_prob = dist.log_prob(minibatch.act)
                    log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                    actor_loss = -(log_prob * minibatch.adv).mean()
                    # calculate loss for critic
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    # calculate regularization and overall loss
                    ent_loss = dist.entropy().mean()
                    loss = actor_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                    
                    actor_loss_avg += actor_loss.item()
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
                actor_losses.append(actor_loss_avg/self.num_gpu)
                vf_losses.append(vf_loss_avg/self.num_gpu)
                ent_losses.append(ent_loss_avg/self.num_gpu)
                losses.append(loss_avg/self.num_gpu)
                    

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        ent_loss_summary_stat = SequenceSummaryStats.from_sequence(ent_losses)

        return A2CTrainingStats(  # type: ignore[return-value]
            loss=loss_summary_stat,
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            ent_loss=ent_loss_summary_stat,
        )
