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
    #Customized DQN class for Udacity deep learning project 1
    def __init__(self, state_size, action_size, seed, use_pixel_input=False):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.drop = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent():
    # The agent interacts with and learns from the banana environment.

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # we use Double DQN for the agent
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
        Q_targets_next = self.dqn_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.dqn_local(states).gather(1, actions)

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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
  
        if use_cuda:
            states, actions, rewards, next_states, dones = states.cuda(), actions.cuda(), rewards.cuda(), next_states.cuda(), dones.cuda()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

