import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib_drqn.common.type_aliases import RNNStates
from skimage.color import rgb2gray

class CustomRecurrentFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim: int = 5, hidden_space: int = 64):
        super().__init__(observation_space, feature_dim)
        self.input_dim = 1 # rgb2gray 3 -> 1 channel observation_space.shape[0]
        self.hidden_space = hidden_space
        self.conv1 = nn.Conv2d(self.input_dim, 16, kernel_size=3, stride=3)  # [N, 1, 96, 96] -> [N, 16, 32, 32]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)  # [N, 16, 32, 32] -> [N, 32, 10, 10]
        self.in_features = 32 * 10 * 10
        
        self.fc1 = nn.Linear(self.in_features, hidden_space)
        self.fc2 = nn.Linear(hidden_space, feature_dim)
        
        self.device = th.device("mps")
        self.img_trans_weights = th.tensor([0.299, 0.587, 0.114], device="mps").view(1, 3, 1, 1)

    def forward(self, x):
        x = self.rgb2gray(x)
        x = th.relu(self.conv1(x))
        x = th.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        return x
    
    def rgb2gray(self, rgb):
        
        gray = th.sum(rgb * self.img_trans_weights, dim=1, keepdim=True)
        return gray
