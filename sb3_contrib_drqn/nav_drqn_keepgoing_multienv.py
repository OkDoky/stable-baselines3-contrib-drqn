import sys
from typing import Dict, List, Tuple
import traceback
import gymnasium as gym
import collections
import random
import numpy as np
import rospy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import sb3_contrib_drqn.nav

from sb3_contrib_drqn.common.env_utils import make_vec_env
from stable_baselines3.common.logger import Logger

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        hidden_space = 256
        feature_dim = action_space
        self.input_dim = 1 
        self.hidden_space = hidden_space
        
        ## data map conv..
        # conv equation. (W - F + 2P) / S + 1 ( W : input size, F : kernel size, P : stride)
        self.conv1 = nn.Conv2d(self.input_dim, 16, kernel_size=4, stride=4)  # [N, 1, 200, 200] -> [N, 16, 50, 50]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)  # [N, 16, 50, 50] -> [N, 32, 16, 16]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # [N, 32, 16, 16] -> [N, 64, 7, 7]
        self.in_features = 64 * 7 * 7

        self.fc1 = nn.Linear(self.in_features, hidden_space)
        self.lstm = nn.LSTM(hidden_space, hidden_space, batch_first=True) 
        self.fc2 = nn.Linear(hidden_space, feature_dim)


    ## lstm input : [batch_size, seq_len, input_size]
    ## conv input : [batch_size, channels, height, width]
    def forward(self, x, h, c):
        # rospy.logwarn(f"[Forward] raw input x : {x.shape}")
        ## (batch_size * seq_len, channels, height, width) reshape
        if len(x.size()) == 4:
            batch_size, seq_len, _h, _w = x.size()
            _c = 1
        elif len(x.size()) == 5:
            batch_size, seq_len, _c, _h, _w = x.size()
        else:
            rospy.logwarn("[Train] something wrong...")
        x = x.view(batch_size * seq_len, _c, _h, _w)

        # rospy.logwarn(f"[Forward] reshape input x (B*S,C,H,W) : {x.shape}")
        x = F.relu(self.conv1(x))
        # rospy.logwarn(f"[Forward] after conv1 x : {x.shape}")
        x = F.relu(self.conv2(x))
        # rospy.logwarn(f"[Forward] after conv2 x : {x.shape}")
        x = F.relu(self.conv3(x))
        # rospy.logwarn(f"[Forward] after conv3 x : {x.shape}")

        ## (batch_size, seq_len, input_size) reshape
        x = x.view(batch_size, seq_len, -1)
        # rospy.logwarn(f"[Forward] after flatten x : {x.shape}")
        x = F.relu(self.fc1(x))
        # rospy.logwarn(f"[Forward] after first linear x : {x.shape}")
        # x = x.unsqueeze(1)  # (batch_size, seq_len, input_size)
        # rospy.logwarn(f"[QNettargets] before lstm x : {x.shape}, h : {h.shape}, c : {c.shape}")
        x, (new_h, new_c) = self.lstm(x, (h,c))
        # rospy.logwarn(f"[QNettargets] x : {x.shape}, new_h : {new_h.shape}, new_c : {new_c.shape}")
        x = self.fc2(x)
        return x, (new_h, new_c)

    def sample_action(self, obs, h,c, epsilon):
        output, (new_h, new_c) = self.forward(obs, h,c)

        if random.random() < epsilon:
            return random.randint(0,1), new_h, new_c
        else:
            return output.squeeze(1).argmax().item(), new_h, new_c
    
    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"
        if training:
            return (torch.zeros([1, batch_size, self.hidden_space]), 
                    torch.zeros([1, batch_size, self.hidden_space]))
        else:
            return (torch.zeros([1, 1, self.hidden_space]), 
                    torch.zeros([1, 1, self.hidden_space]))

class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False, 
                       max_epi_num=100, max_epi_len=500,
                       batch_size=1,
                       lookup_step=None):
        self.random_update = random_update # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if not random_update and self.batch_size > 1:
            sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update: # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if len(episode) >= self.lookup_step: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=0, padding=True)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################           
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None, padding=False) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        if padding:
            padding_size = lookup_step - len(obs)
            obs = np.concatenate(([obs[0]] * padding_size, obs), axis=0)
            action = np.concatenate(([action[0]] * padding_size, action), axis=0)
            reward = np.concatenate(([reward[0]] * padding_size, reward), axis=0)
            next_obs = np.concatenate(([next_obs[0]] * padding_size, next_obs), axis=0)
            done = np.concatenate(([done[0]] * padding_size, done), axis=0)

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, 
          optimizer = None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99,
          writer=None,
          total_steps=0):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)
    # rospy.logwarn(f"[Train] sample size : {observations.shape}, {rewards.shape}, seq len : {seq_len}")

    observations = torch.FloatTensor(observations).view(batch_size, seq_len, 1, 200, 200).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
    next_observations = torch.FloatTensor(next_observations).view(batch_size, seq_len, 1, 200, 200).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size,seq_len,-1)).to(device)

    h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)

    q_target, (new_h, new_c) = target_q_net(next_observations, h_target.to(device), c_target.to(device))
    q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
    targets = rewards + gamma * q_target_max * dones


    h, c = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, (new_h, new_c) = q_net(observations, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss        
    loss = F.smooth_l1_loss(q_a, targets)
    writer.add_scalar("train/loss", loss, total_steps)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='weights/default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # Env parameters
    model_name = "DRQN_POMDP_Random_FOMDP"
    env_name = "DataMap-v0"
    seed = 1
    exp_num = 'SEED'+'_'+str(seed)


    ## set environment
    n_envs = int(rospy.get_param("n_envs", 2))
    env_kwargs = {
        "render_mode": "rgb_array",
    }
    # Set gym environment
    envs = make_vec_env(env_name, n_envs=n_envs, env_kwargs=env_kwargs)

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('tb_logs/'+env_name+"_"+model_name+"_"+exp_num)

    # Set parameters
    batch_size = 8
    learning_rate = 1e-3
    min_epi_num = 8 # Start moment to train the Q network
    episodes = 50000
    print_per_iter = 20
    target_update_period = 64
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 300

    # DRQN param
    random_update = True# If you want to do random update instead of sequential update
    lookup_step = 30 # If you want to do random update instead of sequential update
    max_epi_len = 300
    max_epi_step = max_step

    

    # Create Q functions
    Q = Q_net(state_space=envs.observation_space.shape[0], 
              action_space=envs.action_space.n).to(device)
    Q_target = Q_net(state_space=envs.observation_space.shape[0], 
                     action_space=envs.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start
    
    episode_memory = EpisodeMemory(random_update=random_update, 
                                   max_epi_num=1000, max_epi_len=max_epi_len, 
                                   batch_size=batch_size, 
                                   lookup_step=lookup_step)
    success_buffer = collections.deque(maxlen=10)

    # Train
    epi_count = 0
    total_steps = 0

    obs = envs.reset()
    dones = [False for _ in range(n_envs)]
    episode_record = [EpisodeBuffer() for _ in range(n_envs)]
    h, c = Q.init_hidden_state(batch_size=n_envs, training=True)
    epi_step = [0 for _ in range(n_envs)]

    while epi_count < episodes:
        ## for junk episode..
        

        ##################################################################
        obs_tensor = torch.FloatTensor(obs).to(device)
        actions = []
        for i in range(n_envs):
            a, h[:, i:i+1,:], c[:, i:i+1,:] = Q.sample_action(
                obs_tensor[i].unsqueeze(0).unsqueeze(0).float().to(device),
                h[:, i:i+1,:].to(device), 
                c[:, i:i+1,:].to(device),
                epsilon)
            actions.append(a)
        actions = np.array(actions)

        next_obs, rewards, dones, infos = envs.step(actions)
        total_steps += n_envs

        # make data
        done_masks = [0.0 if d else 1.0 for d in dones]

        for i in range(n_envs):
            episode_record[i].put([obs[i], actions[i], rewards[i]/100.0, next_obs[i], done_masks[i]])
            epi_step[i] += 1

        obs = next_obs
        score += np.mean(rewards)
        score_sum += np.mean(rewards)

        if len(episode_memory) >= min_epi_num:
            train(Q, Q_target, episode_memory, device, 
                    optimizer=optimizer,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    writer=writer,
                    total_steps=total_steps)

            if (total_steps+1) % target_update_period == 0:
                # Q_target.load_state_dict(Q.state_dict()) <- navie update
                for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                        target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

        if any(dones):
            for i, done in enumerate(dones):
                if done:
                    success_buffer.append(infos[i]['is_success'])
        
        ###############################################################

                    epi_count += 1 
                    if epi_count % 10 == 0:
                        success_rate = success_buffer.count(1) / len(success_buffer)
                        success_buffer.clear()
                        writer.add_scalar("train/success_rate", success_rate, epi_count)
        for i in range(n_envs):
            episode_memory.put(episode_record[i])
        
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if epi_count % print_per_iter == 0 and epi_count!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            epi_count, score_sum/print_per_iter, len(episode_memory), epsilon*100))
            score_sum=0.0
            save_model(Q, "weights/"+model_name+"_"+exp_num+"_"+str(epi_count)+'.pth')

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, epi_count)  ## every episode
        score = 0
        
    writer.close()
    envs.close()