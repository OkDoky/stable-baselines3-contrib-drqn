import time
import gymnasium as gym
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from drqn.drqn import DRQN
from drqn.policies import RecurrentDQNPolicy
from common.buffers import RecurrentReplayBuffer
from drqn.extractors import CustomRecurrentFeaturesExtractor



DEVICE=th.device("mps")

## set environment
n_envs = 20
if n_envs > 1:
    env = make_vec_env("CarRacing-v2", n_envs=n_envs, env_kwargs={"domain_randomize": True, "continuous": False})
else:
    env = gym.make("CarRacing-v2", domain_randomize=True, continuous=False, render_mode="rgb_array")

# initial value setup
buffer_size = 5000
batch_size = 64
num_hidden_layers = 5
hidden_size = 64
policy_kwargs = dict(
    features_extractor_class=CustomRecurrentFeaturesExtractor,
    features_extractor_kwargs=dict(feature_dim=5),
    n_lstm_layers=num_hidden_layers,
    lstm_hidden_size=hidden_size,
)

buffer_kwargs = dict(
    # hidden_state_shape=(buffer_size, num_hidden_layers, n_envs, hidden_size), 
    sequence_length=10,
    max_episode_buffer_size=10,
)
st = time.time()
model = DRQN(
    policy=RecurrentDQNPolicy, 
    env=env, 
    buffer_size=buffer_size, 
    learning_starts=200,
    batch_size=batch_size,
    train_freq=(5, "step"),
    policy_kwargs=policy_kwargs, 
    replay_buffer_class=RecurrentReplayBuffer,
    replay_buffer_kwargs=buffer_kwargs,
    verbose=1, 
    tensorboard_log="./tb_logs/", 
    device=DEVICE,
)

model.learn(total_timesteps=1_000_000, tb_log_name="DRQN", log_interval=1)
model.save("drqn_carracing")
print("learning time : %s"%(time.time() - st))
del model  # remove to demonstrate saving and loading
env.close()