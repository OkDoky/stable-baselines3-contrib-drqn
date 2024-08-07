import sys
from collections import deque
import random
from copy import deepcopy
import rospy
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
            ## current_step_transition : tuple(obs, act, rwd, next_obs, done)
            current_step_transition = (obs[env_idx], 
                                       action[env_idx], 
                                       reward[env_idx], 
                                       next_obs[env_idx], 
                                       done[env_idx])
            self.current_episodes[env_idx].append(current_step_transition)

            if done[env_idx]:
                # When episode for a particular env is done, transfer it to the episode buffer
                self.episode_buffers.add_episode(deepcopy(self.current_episodes[env_idx]))
                self.current_episodes[env_idx] = []

    def sample_episodes(self, batch_size):
        """Sample a batch of episodes across all environments."""
        return self.episode_buffers.sample(batch_size, self.sequence_length)
    
    def sample_episode(self, batch_size):
        """Sample a batch of episodes across all environments."""
        return self.episode_buffers.sample_episode(batch_size)
    
    def get_seq_len(self):
        return self.sequence_length
        
class EpisodeBuffer:
    def __init__(self, maxlen=10):
        self.maxlen = maxlen
        self.buffers = deque(maxlen=maxlen)
        self._is_full = False

    def add_episode(self, episode):
        ## buffers : deque(episode)
        ## episode : List[tuple(step transitions)]
        self.buffers.append(episode)
        rospy.logwarn("[Buffers] add episode, episode first index(actions) : %s, type : %s"%(type(self.buffers[-1][0][1]), self.buffers[-1][0][1]))
        self._is_full = bool(len(self.buffers) == self.buffers.maxlen)
        
    def sample(self, batch_size, sequence_length):
        """Sample a batch of episodes, ensuring each is of a fixed sequence length."""
        sampled_episodes = []
        for _ in range(batch_size):
            # Choose a random environment
            env_idx = np.random.randint(len(self.buffers))
            if len(self.buffers[env_idx]) == 0:
                continue
            # Choose a random episode from the environment
            episode_length = len(self.buffers[env_idx])
            # episode_idx = np.random.randint(len(self.buffers[env_idx]))  # int
            # rospy.logerr("[Buffers] episode to list? %s -> %s"%(type(self.buffers[env_idx][episode_idx]), type(list(self.buffers[env_idx][episode_idx]))))
            
            # Trim or pad the episode to the required sequence length
            if episode_length > sequence_length:
                start_idx = np.random.randint(episode_length - sequence_length + 1)
                episode = self.buffers[env_idx][start_idx:start_idx + sequence_length]
            elif episode_length < sequence_length:
                episode = self.buffers[env_idx]
                padding = [self.buffers[env_idx][0]] * (sequence_length - episode_length)  # padding with the first step
                episode = padding + episode
            # rospy.logerr("[Buffers] episode : %s, %s"%(len(episode), episode))
            # rospy.logerr("[Buffers] types : %s"%([type(t) for t in episode]))
            sampled_episodes.append(episode)
        return sampled_episodes

    def sample_episode(self, batch_size):
        """Sample a batch of episodes, ensuring each is of a fixed sequence length."""
        sampled_episodes = []
        env_idx = np.random.randint(len(self.buffers))
        episode_length = len(self.buffers[env_idx])
        
        # Trim or pad the episode to the required sequence length
        if episode_length > batch_size:
            start_idx = np.random.randint(episode_length - batch_size + 1)
            episode = self.buffers[env_idx][start_idx:start_idx + batch_size]
        elif episode_length < batch_size:
            episode = self.buffers[env_idx]
            padding = [self.buffers[env_idx][0]] * (batch_size - episode_length)  # padding with the first step
            episode = padding + episode
        # rospy.logerr("[Buffers] episode : %s, %s"%(len(episode), episode))
        # rospy.logerr("[Buffers] types : %s"%([type(t) for t in episode]))
        # sampled_episodes.append(episode)
        return episode

    def __len__(self) -> int:
        return len(self.buffers)
    
    def is_full(self) -> bool:
        rospy.logwarn("[Buffers] epi buf is full %s"%self._is_full)
        return self._is_full