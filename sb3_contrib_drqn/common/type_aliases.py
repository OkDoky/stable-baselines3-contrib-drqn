from typing import NamedTuple, Tuple

import torch as th
# from stable_baselines3.common.type_aliases import ReplayBufferSamples


class RNNStates(NamedTuple):
    hidden_state: Tuple[th.Tensor, ...]