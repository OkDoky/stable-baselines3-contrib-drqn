import time
import gymnasium as gym
import rospy
import torch as th
import numpy as np
from stable_baselines3.common.monitor import Monitor

from sb3_contrib_drqn.drqn.drqn import DRQN
from sb3_contrib_drqn.drqn.policies import RecurrentDQNPolicy
from sb3_contrib_drqn.drqn.extractors import CustomRecurrentFeaturesExtractor
from sb3_contrib_drqn.common.buffers import RecurrentReplayBuffer
import sb3_contrib_drqn.nav
from sb3_contrib_drqn.common.env_utils import make_vec_env


DEVICE=th.device("mps")
if th.cuda.is_available():
    DEVICE = th.device("cuda")
elif th.backends.mps.is_availeble():
    DEVICE = th.device("mps")
else:
    DEVICE = th.device("cpu")
## set environment
n_envs = int(rospy.get_param("n_envs", 2))
env_kwargs = {
    "render_mode": "rgb_array",
}
if n_envs > 1:
    env = make_vec_env("DataMap-v0", n_envs=n_envs, env_kwargs=env_kwargs)
else:
    env = gym.make("DataMap-v0", ns="r1",**env_kwargs)

# load model
model = DRQN.load("weights/drqn_data_map", env=env, device=DEVICE)

# eval code
n_episodes = 100
episode_rewards = []

for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    hidden_state = None
    while not done:
        if isinstance(obs, dict):
            obs_tensor = {key: th.tensor(np.array(value), dtype=th.float32, device=DEVICE) for key, value in obs.items()}
        else:
            obs_tensor = th.tensor(np.array([np.array(o) for o in obs]), dtype=th.float32, device=DEVICE)
        if hidden_state is not None:
            hidden_state = hidden_state.hidden_state
        action, hidden_state = model.predict(obs_tensor, state=hidden_state, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        episode_reward += reward

    episode_rewards.append(episode_reward)
    print(f"Episode {ep+1}: Reward = {episode_reward}")

avg_reward = sum(episode_rewards) / n_episodes
print(f"Average Reward over {n_episodes} episodes: {avg_reward}")

env.close()