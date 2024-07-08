from __future__ import annotations
from typing import Any, Union, Optional, Tuple, ClassVar, Dict, Type, TypeVar

import numpy as np
from gymnasium import spaces
import torch as th
from torch.nn import functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib_drqn.drqn.policies import RecurrentDQNPolicy, RecurrentQNetwork
from sb3_contrib_drqn.common.buffers import RecurrentReplayBuffer
from sb3_contrib_drqn.common.type_aliases import RNNStates

SelfDRQN = TypeVar("SelfDRQN", bound="DRQN")

class DRQN(DQN):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "RecurrentDQNPolicy": RecurrentDQNPolicy,
    }
    q_net: RecurrentQNetwork
    q_net_target: RecurrentQNetwork
    policy: RecurrentDQNPolicy
    
    def __init__(
        self,
        policy: Union[str, Type[RecurrentDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[RecurrentReplayBuffer]] = RecurrentReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = th.device("mps"),
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        self.lstm_states = None
    
    def _setup_model(self) -> None:
        super()._setup_model()
        

    def predict(self, observation: np.ndarray, 
                state: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
                episode_start: Optional[np.ndarray] = None, 
                deterministic: bool = False) -> Tuple[np.ndarray, RNNStates]:
        observation = th.as_tensor(observation).float().to(self.device)
        
        if state is None:
            state = self.policy.get_lstm_states(batch_size=1)
        else:
            (h, c) = state
            state = RNNStates(
                hidden_state=(
                    h.clone().detach().to(self.device), 
                    c.clone().detach().to(self.device)
                )
            )
        
        actions = self.policy._predict(observation, state, deterministic)
        # print("[DRQN] policy predicted actions : ", actions)
        return actions.cpu().numpy(), state

    def learn(
        self: SelfDRQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDRQN:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.replay_buffer.episode_buffers.is_full():
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        seq_len = self.replay_buffer.get_seq_len()
        for _ in range(gradient_steps):
            # Sample episode buffer
            samples = self.replay_buffer.sample_episodes(batch_size)  # type: ignore[union-attr]
            
            _observations = []
            _actions = []
            _rewards = []
            _next_observations = []
            _dones = []
            for i in range(self.batch_size):
                for seq in range(seq_len):
                    _observations.append(samples[i][seq][0])
                    _actions.append(samples[i][seq][1])
                    _rewards.append(samples[i][seq][2])
                    _next_observations.append(samples[i][seq][3])
                    _dones.append(samples[i][seq][4])
            print("[DRQN] sampled obs shape : ")
            _observations = th.FloatTensor(np.array(_observations).reshape(self.batch_size, seq_len, -1)).to(self.device)
            _actions = th.LongTensor(np.array(_actions).reshape(self.batch_size, seq_len, -1)).to(self.device)
            _rewards = th.FloatTensor(np.array(_rewards).reshape(self.batch_size, seq_len, -1)).to(self.device)
            _next_observations = th.FloatTensor(np.array(_next_observations).reshape(self.batch_size, seq_len, -1)).to(self.device)
            _dones = th.FloatTensor(np.array(_dones).reshape(self.batch_size, seq_len, -1)).to(self.device)
            
            with th.no_grad():
                # Compute the next Q-values using the target network
                h_target, c_target = self.q_net_target.get_lstm_states(batch_size=self.batch_size)
                next_q_values, _, _ = self.q_net_target(_next_observations, (h_target, c_target))
                next_q_values = next_q_values.max(dim=1).reshape(-1, 1)
                target_q_values = _rewards + (1 - _dones) * self.gamma * next_q_values
            
            # Get current Q-values estimates
            h, c = self.q_net.get_lstm_states(batch_size=self.batch_size)
            current_q_values, _, _ = self.q_net(_observations, h, c)
            current_q_values = th.gather(current_q_values, dim=1, index=_actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            
            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, self.lstm_states = self.predict(self._last_obs, state=self.lstm_states.hidden_state, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def _store_transition(
        self,
        replay_buffer: RecurrentReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list,
    ) -> None:
        """
        Store a transition in the replay buffer.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.unnormalize_obs(new_obs)
        else:
            new_obs_ = new_obs

        # Initialize lstm_states if they are None
        if self.lstm_states is None:
            self.lstm_states = self.policy.get_lstm_states(batch_size=1)
        # Store the transition in the replay buffer
        replay_buffer.add(
            self._last_obs,
            new_obs_,
            buffer_action,
            reward,
            done,
            infos
        )

        # Update the last observation
        self._last_obs = new_obs
        _, self.lstm_states = self.policy.forward(th.as_tensor(new_obs).float().to(self.device), self.lstm_states)
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: RecurrentReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)