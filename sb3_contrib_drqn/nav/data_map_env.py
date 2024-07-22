import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib_drqn.nav.conditions import ResultCondition
from sb3_contrib_drqn.nav.simulator_connector import SimulatorHandler

class DataMapEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        ns: str,
        # n_envs: int,
        max_steps_per_episode: int = 300,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.ns = ns
        # self.n_envs = n_envs
        self.max_steps_per_episode = max_steps_per_episode
        self.log_history_lenght = 10
        
        self.result_history = [0] * 3
        self.mean_reward = [0, 0]
        self.last_mean_reward = 0
        self.step_time = [0, 0]
        self._steps_current_episode = 0
        self._episode = 0
        
        # self.action_space = spaces.MultiDiscrete([3, 3, 2])  # linear acc (-0.5, 0.0, 0.5), angular vel (-0.5, 0.0, 0.5), breaking (true/false)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 200, 200), dtype=np.uint8)
        self.simulator = SimulatorHandler(ns)

    def step(self, action):
        start_time = time.time()
        self.simulator.send_action(action)
        self._steps_current_episode += 1
        obs, reward, done, info = self.simulator.get_transitions()
        
        if self._steps_current_episode >= self.max_steps_per_episode:
            done = True
            info["result_condition"] = ResultCondition.TIMEOUT
            info["is_success"] = 0
        
        self.step_time[1] += 1
        self.mean_reward[1] += 1
        self.mean_reward[0] += reward
        
        if done:
            if sum(self.result_history) >= self.log_history_lenght:
                mean_reward = self.mean_reward[0] / self.mean_reward[1]
                diff = round(mean_reward - self.last_mean_reward, 3)
                result_log = "[Last %d Episodes]\t"%(self.log_history_lenght) \
                    + "[Success] : %s\t"%(self.result_history[0]) \
                    + "[Collision] : %s\t"%(self.result_history[1]) \
                    + "[Timeout] : %s\t"%(self.result_history[2])
                print(result_log)
                self.result_history = [0] * 3
                self.step_time = [0, 0]
                self.last_mean_reward = mean_reward
                self.mean_reard = [0, 0]
            
            self.result_history[int(info["result_condition"])] += 1
            self._steps_current_episode = 0
        self.step_time[0] += time.time() - start_time            
        return obs, reward, done, None, info

    def reset(self, seed=None, options=None):
        self._episode += 1
        self.simulator.reset()
        observation = self.simulator.get_observation()
        info = {}
        return observation, info

    def close(self):
        pass