import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # interpolation parameter for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 4        # how often to update the network

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    #Customized DQN class for Udacity deep reinforcement learning project 1
    def __init__(self, state_size, action_size, seed):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def conv(in_channels, out_channels, kernel_size, stride = 1, padding = 1, batch_norm = True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
    layers.append(conv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

class CNNDQN(nn.Module):
    #Customized CNN and DQN class for Udacity deep reinforcement learning project 1
    #The class is referenced from the projects in Udacity deep learning course.
    def __init__(self, state_size, action_size, seed):
        super(CNNDQN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = conv(3, 4, 3, batch_norm = False)
        self.conv2 = conv(4, 8, 3)
        self.conv3 = conv(8, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(800, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent():
    # The agent interacts with and learns from the banana environment.

    def __init__(self, state_size, action_size, visual_input, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # we use Double DQN for the agent
        if visual_input:
            self.dqn_local = CNNDQN(state_size, action_size, seed)
            self.dqn_target = CNNDQN(state_size, action_size, seed)
        else:
            self.dqn_local = DQN(state_size, action_size, seed)
            self.dqn_target = DQN(state_size, action_size, seed)
            
        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)
        if use_cuda:
            dqn_local, dqn_target = dqn_local.cuda(), dqn_target.cuda()
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.step_counter = 0
    
    def step(self, state, action, reward, next_state, done):
        # Record the experience, update the network when running enough step.
        self.memory.add(state, action, reward, next_state, done)

        self.step_counter = (self.step_counter + 1) % UPDATE_EVERY
        if self.step_counter == 0:
            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        #Returns action for given state as per current policy.

        state = torch.from_numpy(state).float().unsqueeze(0)
        if use_cuda:
            state = state.cuda()
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        #Update value parameters using given batch of experience tuples.

        states, actions, rewards, next_states, dones = experiences
        self.dqn_local.train()
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.dqn_target(next_states).detach().max(1)[0]

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        #                       64, 4              64
        Q_expected = self.dqn_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update target network
        self.soft_update(self.dqn_local, self.dqn_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        #Soft update model parameters.
        #θ_target = τ * θ_local + (1 - τ) * θ_target

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, path):
        torch.save(self.dqn_local.state_dict(), path)
    
    def load(self, path):
        self.dqn_local.load_state_dict(torch.load(path))
        self.dqn_target = self.dqn_local

class ReplayBuffer:
    #Fixed-size buffer to store experience tuples.

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        #Randomly sample a batch of experiences from memory.
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor([e.state for e in experiences if e is not None])
        actions = torch.LongTensor([e.action for e in experiences if e is not None])
        rewards = torch.FloatTensor([e.reward for e in experiences if e is not None])
        next_states = torch.FloatTensor([e.next_state for e in experiences if e is not None])
        dones = torch.FloatTensor([e.done for e in experiences if e is not None])
  
        if use_cuda:
            states, actions, rewards, next_states, dones = states.cuda(), actions.cuda(), rewards.cuda(), next_states.cuda(), dones.cuda()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

