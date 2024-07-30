
from gymnasium.envs.registration import register
from sb3_contrib_drqn.nav.data_map_env import DataMapEnv
register(
     id="DataMap-v0",
     entry_point="sb3_contrib_drqn.nav:DataMapEnv",
     max_episode_steps=600,  ## 60 sec
)