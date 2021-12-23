import Models
import Memory
import numpy as np
import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class spiked_model:

    def __init__(self, obs, action, buffer_size, batch_size):
        self.obs = obs
        self.action = action
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.gamma = 0.99
        self.tau = 1e-3
        self.update = 0
        self.memory = Memory.Expreince_Replay(self.buffer_size)

        self.q_network = Models.Spiked_DQN(obs, action)
        self.q_target = Models.Spiked_DQN(obs, action)

        self.optim = optim.Adam(self.q_network.parameters(),  lr=5e-4, betas=(0.9, 0.999))
    
    def act(self, state, eps):

        if np.random.random_sample() > eps:
            self.q_network.eval()
            with torch.no_grad():
                action_vals, _ = self.q_network(state)
            
            self.q_network.train()

            action_vals = action_vals.sum(dim=0)
            action = np.argmax(action_vals.cpu().data.numpy())
            return action
        
        else:
            return np.random.randint(self.action)
    
    def learn(self):
        state, action, reward, next_state, done = self.get_mem()
        self.update += 1
        
        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        _, mem_rec = self.q_network(state)
        q_values = mem_rec.sum(dim=0)
        _, mem_rec = self.q_network(next_state)
        next_q_values = mem_rec.sum(dim=0)
        _, mem_rec = self.q_target(next_state)
        next_state_q_values = mem_rec.sum(dim=0)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_state_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + (self.gamma * next_q_value * (1-done))

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def push_mem(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    
    def get_mem(self):
        return self.memory.sample(self.batch_size)


    def update_target(self):
        self.q_target.load_state_dict(self.q_network.state_dict())
    
    def is_update(self):
        if len(self.memory) > self.batch_size:
            self.learn()

