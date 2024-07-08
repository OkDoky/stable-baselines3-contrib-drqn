import sys
from collections import deque
import random
from typing import Union, Tuple, Optional, Any, List, Dict
import numpy as np
import torch as th

from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer

class RecurrentReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        sequence_length: int = 30,
        max_episode_buffer_size: int = 64,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.episode_buffers = EpisodeBuffer(maxlen=max_episode_buffer_size)
        self.current_episodes = [[] for _ in range(self.n_envs)]
        self.sequence_length = sequence_length
        
    def add(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Extend the add method to handle multiple environments."""
        for env_idx in range(self.n_envs):
            self.current_episodes[env_idx].append((obs[env_idx], action[env_idx], reward[env_idx], next_obs[env_idx], done[env_idx]))

            if done[env_idx]:
                # When episode for a particular env is done, transfer it to the episode buffer
                self.episode_buffers.add_episode(self.current_episodes[env_idx].copy())
                self.current_episodes[env_idx] = []

    def sample_episodes(self, batch_size):
        """Sample a batch of episodes across all environments."""
        return self.episode_buffers.sample(batch_size, self.sequence_length)

    def get_seq_len(self):
        return self.sequence_length
        
class EpisodeBuffer:
    def __init__(self, maxlen=10):
        self.maxlen = maxlen
        self.buffers = deque(maxlen=maxlen)
        self._is_full = False

    def add_episode(self, episode):
        self.buffers.append(episode)
        
    def sample(self, batch_size, sequence_length):
        """Sample a batch of episodes, ensuring each is of a fixed sequence length."""
        sampled_episodes = []
        for _ in range(batch_size):
            # Choose a random environment
            env_idx = np.random.randint(len(self.buffers))
            if len(self.buffers[env_idx]) == 0:
                continue
            # Choose a random episode from the environment
            episode_idx = np.random.randint(len(self.buffers[env_idx]))
            episode = self.buffers[env_idx][episode_idx]
            # Trim or pad the episode to the required sequence length
            if len(episode) > sequence_length:
                start_idx = np.random.randint(len(episode) - sequence_length + 1)
                episode = episode[start_idx:start_idx + sequence_length]
            elif len(episode) < sequence_length:
                padding = [episode[0]] * (sequence_length - len(episode))  # padding with the first step
                episode = padding + episode
            sampled_episodes.append(episode)
        return sampled_episodes

    def __len__(self) -> int:
        return len(self.buffers)
    
    def is_full(self) -> bool:
        return self._is_full