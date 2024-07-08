import torch as th
import torch.nn as nn
from typing import Optional, Tuple, Type, Dict, Any, Callable, List, Union

from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from sb3_contrib_drqn.common.type_aliases import RNNStates
from sb3_contrib_drqn.drqn.extractors import CustomRecurrentFeaturesExtractor

class RecurrentQNetwork(BasePolicy):
    features_extractor: CustomRecurrentFeaturesExtractor
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Union[Callable, float] = 1e-3,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        net_arch: Optional[List[int]] = None,
        device: Union[th.device, str] = "auto",
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BasePolicy] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(RecurrentQNetwork, self).__init__(
            observation_space, 
            action_space, 
            net_arch,
            device,
            features_extractor_class, 
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        if features_extractor_class is not None:
            self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs).to(self.device)
            self.features_dim = self.features_extractor.features_dim
            # print("[RecurrentQNetWork] featrues dim : ", self.features_dim)
        else:
            self.features_dim = observation_space

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.features_dim, 
                            hidden_size=self.lstm_hidden_size, 
                            num_layers=self.n_lstm_layers, 
                            batch_first=True)

        # Define the rest of the Q-network
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            activation_fn(),
            nn.Linear(lstm_hidden_size, action_space.n)
        )

    def forward(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        # # obs: (batch_size, seq_len, input_dim)
        # if self.features_extractor is not None:
        #     print("[RecurrentQNetWork] before extractor obs shape :", obs.shape)
        #     obs = self.features_extractor(obs)
        #     print("[RecurrentQNetWork] after extractor obs shape :", obs.shape)
        # print("[RecurrentQNetWork] obs shape :", obs.shape, ", lstm state : ", lstm_states[0].shape, lstm_states[1].shape)
        lstm_out, lstm_states = self.lstm(obs, lstm_states)
        q_values = self.fc(lstm_out)
        return q_values, lstm_states

    def _predict(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, th.Tensor], deterministic: bool = True) -> th.Tensor:
        q_values, _ = self.forward(obs, lstm_states)
        actions = q_values.argmax(dim=1) if deterministic else th.multinomial(th.softmax(q_values, dim=1), 1).squeeze()
        return actions

    def get_q_values(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        q_values, _ = self.forward(obs, lstm_states)
        return q_values

    def get_lstm_states(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        # Initialize LSTM states
        h_n = th.zeros((self.n_lstm_layers, batch_size, self.lstm_hidden_size)).to(self.device)
        c_n = th.zeros((self.n_lstm_layers, batch_size, self.lstm_hidden_size)).to(self.device)
        return h_n, c_n

class RecurrentDQNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch: Optional[list] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomRecurrentFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super(RecurrentDQNPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs).to(self.device)
        self.features_dim = self.features_extractor.features_dim

        # Define the Recurrent Q-network
        self.q_net = RecurrentQNetwork(
            observation_space, 
            self.action_space, 
            lstm_hidden_size=lstm_hidden_size, 
            n_lstm_layers=n_lstm_layers, 
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )
        self.q_net_target = RecurrentQNetwork(
            observation_space, 
            self.action_space, 
            lstm_hidden_size=lstm_hidden_size, 
            n_lstm_layers=n_lstm_layers, 
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )

    def forward(self, obs: th.Tensor, lstm_states: RNNStates) -> Tuple[th.Tensor, RNNStates]:
        # print("[RecurrentDQNPOlicy] obs : ", obs.shape)
        features = self.extract_features(obs, self.features_extractor)
        # print("[RecurrentDQNPOlicy] after extractor obs : ", obs.shape)
        features = features.unsqueeze(0)  # Add batch dimension for LSTM

        q_values, (h_n, c_n) = self.q_net(features, lstm_states.hidden_state)
        q_values = q_values.squeeze(0)  # Remove batch dimension

        return q_values, RNNStates(hidden_state=(h_n, c_n))

    def _predict(self, obs: th.Tensor, lstm_states: RNNStates, deterministic: bool = True) -> th.Tensor:
        q_values, _ = self.forward(obs, lstm_states)
        actions = q_values.argmax(dim=1) if deterministic else th.multinomial(th.softmax(q_values, dim=1), 1).squeeze()
        return actions

    def get_lstm_states(self, batch_size: int) -> RNNStates:
        # Initialize LSTM states
        h_n = th.zeros((self.n_lstm_layers, batch_size, self.lstm_hidden_size)).to(self.device)
        c_n = th.zeros((self.n_lstm_layers, batch_size, self.lstm_hidden_size)).to(self.device)
        return RNNStates(hidden_state=(h_n, c_n))

    def extract_features(self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)