#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DDPG is based on https://github.com/higgsfield/RL-Adventure-2 #

import copy
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import PER_buffer
from queue import Queue
# USE CUDA GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=0.04, min_sigma=0, decay_period=500000):  # 1000000
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space

    def sample(self, action, step=0):
        sigma = self.get_sigma(step)
        noise_v = np.random.normal(loc=0, size= 1 , scale=sigma)  # * sigma
        noise_w = np.random.normal(loc=0, size= 1 , scale=sigma*5)
        noise = np.append(noise_v, noise_w)
        return noise
    
    def get_sigma(self, step=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step * 1.0 / self.decay_period)
        return sigma
    
class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, max_lin_vel, max_ang_vel, obs_state, robot_state, lstm_hidden, init_w=3e-3):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(hidden_size, lstm_hidden , batch_first=True)
        self.lstm_hidden_dim = lstm_hidden
        self.num_inputs = num_inputs
        self.last_h = torch.zeros(1, 1, self.lstm_hidden_dim).to(device)
        self.last_c = torch.zeros(1, 1, self.lstm_hidden_dim).to(device)
        self.q = Queue(10)
        self.c = Queue(10)
        self.forward_model = nn.Sequential(
            nn.Linear(lstm_hidden , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
        )

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
    def forward(self, obs, hist_obs, hist_act, hist_seg_len, lea=True, change=False):
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        x = hist_obs             
         
        h = torch.zeros(1, x.shape[0], self.lstm_hidden_dim).to(device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden_dim).to(device)

        if not lea :
            h = self.last_h
            c = self.last_c

        x = self.feature_layer(x)  
        #  print(x.shape)  #torch.Size([64, 10, 256])
        x, (lstm_hidden_state, lstm_cell_state) = self.lstm(x,(h,c))

        if x.shape[0] == 1:
            if change:
                while not self.q.empty():
                    self.q.get()
                    self.c.get()
            self.q.put(lstm_hidden_state)
            self.c.put(lstm_cell_state)
            if self.q.full():
                self.last_h = self.q.get()
                self.last_c = self.c.get()

        hist_out = torch.gather(x, 1,(tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long()).squeeze(1)
        action = self.forward_model(hist_out)

        action[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel

        return action
        # print(hist_obs.shape)               #torch.Size([64, 10, 43])
        # print(x.shape)                      #torch.Size([64, 10, 256])
        # print(tmp_hist_seg_len.shape)       #torch.Size([64])
        # print((tmp_hist_seg_len - 1).view(-1, 1).shape)  #torch.Size([64, 1])
        # print((tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).shape) #torch.Size([64, 256])
        # print((tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long().shape) #torch.Size([64, 1 , 256])
        # print(torch.gather(x, 1,(tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long()).shape)  
        # torch.Size([64, 1 , 256])
        # print(torch.gather(x, 1,(tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long()).squeeze(1).shape)
        #torch.Size([64, 256])

        # print(tmp_hist_seg_len - 1)
        # print((tmp_hist_seg_len - 1).view(-1, 1))
        # print((tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim))
        # print((tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long())
        
        # print(x)
        # print(torch.gather(x, 1,(tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long()))
  
        # print(hist_out)
        # x = obs
        # x = self.post_layer(x)
        # extracted_memory = hist_out
        # x = torch.cat([extracted_memory, x], dim=-1)
        # action = self.forward_model(x)
        
        # Give two action, linear vel: {0,1}, angular vel: {-1,1}


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, obs_state, robot_state, lstm_hidden, init_w=3e-3):
        super(Critic, self).__init__()
        self.lstm_hidden_dim = lstm_hidden
        self.num_inputs = num_inputs
        self.lstm = nn.LSTM(hidden_size, lstm_hidden , batch_first=True)
        self.lstm_hidden_dim = lstm_hidden
        self.forward_model = nn.Sequential(
            nn.Linear(lstm_hidden + 16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
        )

        self.action_layer = nn.Sequential(
            nn.Linear(num_actions, 16),
            nn.ReLU(),
        )
    def forward(self, state, action, hist_obs, hist_act, hist_seg_len):
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        x = hist_obs
        h = torch.zeros(1, x.shape[0], self.lstm_hidden_dim).to(device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden_dim).to(device)
        x = self.feature_layer(x)
        x, (lstm_hidden_state, lstm_cell_state) = self.lstm(x,(h,c))
        hist_out = torch.gather(x, 1,(tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.lstm_hidden_dim).unsqueeze(1).long()).squeeze(1)
        action =  self.action_layer(action)
        # x = self.post_layer(state)
        # extracted_memory = hist_out
        # x = torch.cat([extracted_memory, x, action], dim=-1)
        x = torch.cat([hist_out, action], dim=-1)
        value = self.forward_model(x)
        return value


class Agent:
    """Main DDPG agent that extracts experiences and learns from them"""
    """
    state_size:    State dimension 363
    action_size:                    2
    hidden_size:                   256
    actor_learning_rate:          0.0001
    critic_learning_rate:         0.001
    batch_size                     64
    buffer_size                   1000000
    discount_factor:  gamma       0.99
    softupdate_coefficient        0.001
    max_lin_vel                   0.22
    max_ang_vel                    2
    """

    def __init__(self, state_size, action_size, hidden_size, actor_learning_rate, critic_learning_rate, batch_size,
                 buffer_size, discount_factor, softupdate_coefficient, max_lin_vel, max_ang_vel):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.loss_function = nn.MSELoss()
        self.tau = softupdate_coefficient
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        
        self.obs_state = 37
        self.robot_state = 6
        self.lstm_hidden = 256
        self.last_ep = -1
        self.lea = False

        # self.hist_obs = np.zeros([batch_size, 10, self.obs_dim])
        # self.hist_act = np.zeros([batch_size, 10, self.act_dim])
        # self.hist_obs_len = np.zeros(batch_size)
        # Actor network
        self.actor_local = Actor(self.state_size, self.action_size, self.hidden_size, self.max_lin_vel,
                                 self.max_ang_vel, self.obs_state, self.robot_state, self.lstm_hidden).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.hidden_size, self.max_lin_vel,
                                  self.max_ang_vel, self.obs_state, self.robot_state, self.lstm_hidden).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic network
        self.critic_local = Critic(self.state_size, self.action_size, self.hidden_size, self.obs_state, self.robot_state, self.lstm_hidden).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.hidden_size, self.obs_state, self.robot_state, self.lstm_hidden).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

        # Noise process
        # self.noise = OUNoise(self.action_size)
        self.noise = GaussianExploration(self.action_size)
        # Replay memory
        self.memory = PER_buffer.ReplayBuffer(self.buffer_size)

        # Update target network with hard updates
        self.hard_update(self.critic_target, self.critic_local)
        self.hard_update(self.actor_target, self.actor_local)

    def step(self, state, action, reward, next_state, done):
        """
        Saves an experience in the replay memory to learn from using random sampling.
        @Param:
        1. state: current state, S.
        2. action: action taken based on current state.
        3. reward: immediate reward from state, action.
        4. next_state: next state, S', from action, a.
        5. done: (bool) has the episode terminated?
        Exracted version for trajectory used in calculating the value for an action, a."""

        self.memory.add(state, action, reward, next_state, done)

    def reset(self):
        """Resets the noise process to mean"""
        self.noise.reset()

    def act(self, state, o_buff, a_buff, o_buff_len, step, ep, add_noise=True):
        """
        Returns a deterministic action given current state.
        @Param:
        1. state: current state, S.
        2. add_noise: (bool) add bias to agent, default = True (training mode)
        """
        self.lea = False
        change = False
        if ep > self.last_ep:
            self.lea = True
            change = True
        if o_buff_len != 10:
            o = np.zeros([o_buff_len, self.state_size])
            o = o_buff[:o_buff_len]
            h_o = torch.tensor(o).view(1, o.shape[0], o.shape[1]).float().to(device)
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # typecast to torch.Tensor
        self.actor_local.eval()  # set in evaluation mode
        with torch.no_grad():  # reset gradients
            action = self.actor_local(state, h_o, h_a, h_l, self.lea, change).cpu().data.numpy()  # deterministic action based on Actor's forward pass.
        self.actor_local.train()  # set training mode

        if add_noise:
            # print(action)
            noise = self.noise.sample(action, step)
            # print(noise)
            action += noise
            # print(action)
       
        # Set upper and lower bound of action spaces
        action[0, 0] = np.clip(action[0, 0], 0.0, self.max_lin_vel)
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)

        if step <= 2500 and add_noise:
            action = np.array([[np.random.uniform(0 , 0.22) , np.random.uniform(-2.0 , 2.0)]])

        self.last_ep = ep

        return action

    def learn(self, ep):
        """
        Learn from a set of experiences picked up from a random sampling of even frequency (not prioritized)
        of experiences when buffer_size = MINI_BATCH.
        Updates policy and value parameters accordingly
        @Param:
        1. experiences: (Tuple[torch.Tensor]) set of experiences, trajectory, tau. tuple of (s, a, r, s', done)
        2. gamma: immediate reward hyper-parameter, 0.99 by default.
        """
        data = self.memory.sample_batch_with_history(self.batch_size)
        states, actions, rewards, next_states, dones = (data[key] for key in ['obs', 'act', 'rew', 'obs2', 'done'])
        hist_obs, hist_act, hist_obs2, hist_act2, hist_obs_len, hist_obs2_len = (data[key] for key in ['hist_obs', 'hist_act', 'hist_obs2', 'hist_act2', 'hist_obs_len','hist_obs2_len'])
        # self.hist_obs = hist_obs

        hist_obs = torch.FloatTensor(hist_obs).to(device)
        hist_act = torch.FloatTensor(hist_act).to(device)
        hist_obs2 = torch.FloatTensor(hist_obs2).to(device)
        hist_act2 = torch.FloatTensor(hist_act2).to(device)
        hist_obs_len = torch.FloatTensor(hist_obs_len).to(device)
        hist_obs2_len = torch.FloatTensor(hist_obs2_len).to(device)

        state = torch.FloatTensor(states).to(device)
        next_state = torch.FloatTensor(next_states).to(device)
        action = torch.FloatTensor(actions).to(device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)

        actor_loss = self.critic_local(state, self.actor_local(state, hist_obs, hist_act, hist_obs_len), hist_obs, hist_act, hist_obs_len)
        actor_loss = -actor_loss.mean()

        next_action = self.actor_target(next_state, hist_obs2, hist_act2, hist_obs2_len)
        target_Q = self.critic_target(next_state, next_action.detach(), hist_obs2, hist_act2, hist_obs2_len)
        expected_Q = reward + (1.0 - done) * self.discount_factor * target_Q
        expected_Q = torch.clamp(expected_Q, -np.inf, np.inf)

        # Remedy: Change action shape from (X, 1, 2) to (1, 2) for concat
        # Cause of action changing from (1, 2) after converting to FloatTensor
        # becoming (X, 1, 2) unknown
        action = torch.squeeze(action, 1)

        Q = self.critic_local(state, action, hist_obs, hist_act, hist_obs_len)
        critic_loss = self.loss_function(Q, expected_Q.detach())

        # Update Actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target network with soft updates
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters. Copies model τ every experience.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + local_param.data * self.tau
            )

    def hard_update(self, target_model, local_param):
        for target_param, param in zip(target_model.parameters(), local_param.parameters()):
            target_param.data.copy_(param.data)

    def save_actor_model(self, outdir, name):
        torch.save(self.actor_target.state_dict(), outdir + '/' + str(name))

    def save_critic_model(self, outdir, name):
        torch.save(self.critic_target.state_dict(), outdir + '/' + str(name))

    def load_models(self, actor_outdir, critic_outdir):
        self.actor_local.load_state_dict(torch.load(actor_outdir))
        self.critic_local.load_state_dict(torch.load(critic_outdir))
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
