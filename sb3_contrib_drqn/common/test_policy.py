import torch as th
import torch.nn as nn
from typing import Optional, Tuple, Type

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import Schedule

from sb3_contrib_drqn.common.type_aliases import RNNStates

class RecurrentQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super(RecurrentQNetwork, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, n_lstm_layers, batch_first=True)

        # Define the rest of the Q-network
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            activation_fn(),
            nn.Linear(lstm_hidden_size, action_dim)
        )

    def forward(self, x: th.Tensor, hidden_state: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        # x: (batch_size, seq_len, input_dim)
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc(lstm_out)
        return q_values, hidden_state

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
        *args,
        **kwargs,
    ):
        super(RecurrentDQNPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        # Define the Recurrent Q-network
        print("[RecurrentDQNPolicy] action space : %s"%action_space)
        self.q_net = RecurrentQNetwork(self.features_dim, self.action_space, lstm_hidden_size, n_lstm_layers, activation_fn)
        self.q_net_target = RecurrentQNetwork(self.features_dim, self.action_space, lstm_hidden_size, n_lstm_layers, activation_fn)

    def forward(self, obs: th.Tensor, lstm_states: RNNStates) -> Tuple[th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        features = features.unsqueeze(0)  # Add batch dimension for LSTM
        print("[RDQNPolicy] feature s : ", features.shape, ", hidden s : ", lstm_states.hidden_state.shape)
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

# Example usage:
# policy = RecurrentDQNPolicy(observation_space, action_space, lr_schedule)
# lstm_states = policy.get_lstm_states(batch_size=1)
# obs = th.tensor(observation).float().to(policy.device)
# action = policy._predict(obs, lstm_states)