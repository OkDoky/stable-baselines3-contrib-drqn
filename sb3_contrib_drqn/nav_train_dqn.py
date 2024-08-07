import time
import gymnasium as gym
import rospy
import torch as th
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
# from sb3_contrib_drqn.drqn.policies import RecurrentDQNPolicy
from sb3_contrib_drqn.drqn.extractors import CustomRecurrentFeaturesExtractor
from sb3_contrib_drqn.common.buffers import RecurrentReplayBuffer
from sb3_contrib_drqn.common.env_utils import make_vec_env
import  sb3_contrib_drqn.nav

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
# if n_envs > 1:
#     env = make_vec_env("DataMap-v0", n_envs=n_envs, env_kwargs=env_kwargs)
# else:
#     env = gym.make("DataMap-v0", ns="r1", **env_kwargs, )

env = make_vec_env("DataMap-v0", n_envs=n_envs, env_kwargs=env_kwargs)

# initial value setup
buffer_size = 5000
batch_size = 64
num_hidden_layers = 5
hidden_size = 16
policy_kwargs = dict(
    features_extractor_class=CustomRecurrentFeaturesExtractor,
    features_extractor_kwargs=dict(feature_dim=5),
)

buffer_kwargs = dict(
    # hidden_state_shape=(buffer_size, num_hidden_layers, n_envs, hidden_size), 
    sequence_length=10,
    max_episode_buffer_size=1,
)
st = time.time()
model = DQN(
    policy=DQNPolicy, 
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

total_timesteps_ = int(rospy.get_param("/total_timesteps", 300_000))
model.learn(total_timesteps=total_timesteps_, tb_log_name="DRQN", log_interval=1)
model.save("weights/drqn_data_map")
print("learning time : %s"%(time.time() - st))
del model  # remove to demonstrate saving and loading
env.close()