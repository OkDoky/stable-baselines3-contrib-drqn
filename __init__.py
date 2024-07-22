import os

from sb3_contrib_drqn.drqn import DRQN

# # Read version from file
# version_file = os.path.join(os.path.dirname(__file__), "version.txt")
# with open(version_file) as file_handler:
#     __version__ = file_handler.read().strip()

__all__ = [
    "DRQN",
]
from gymnasium.envs.registration import register

register(
     id="DataMap-v0",
     entry_point="sb3_contrib_drqn.nav.data_map_env:CustomEnv",
     max_episode_steps=600,  ## 60 sec
)