import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import snntorch.functional as SF
import numpy as np



class DQN(nn.Module):
    def __init__(self, obs,  action):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(obs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action)
        )
    
    def forward(self, x, goal):
        x = torch.FloatTensor(x)
        x = torch.FloatTensor(goal)
        x = torch.cat([x, goal], 1)
        x = self.model(x)

        return x

class Spiked_DQN(nn.Module):
    def __init__(self, obs, action):
        super(Spiked_DQN, self).__init__()

        
        self.beta = 0.85
        self.n_steps = 25
        self.input_layer = nn.Linear(obs, 128)
        self.lif1 = snn.Leaky(beta=self.beta)
        self.hidden1 = nn.Linear(128, 128)
        self.lif2 = snn.Leaky(beta=self.beta)
        self.output_layer = nn.Linear(128, action)
        self.lifo = snn.Leaky(beta=self.beta)
    
    def forward(self, x):
        x = torch.FloatTensor(x)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        memo = self.lifo.init_leaky()

        spko_rec = []
        memo_rec = []

        for step in range(self.n_steps):
            cur1 = self.input_layer(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.hidden1(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.output_layer(spk2)
            spko, memo = self.lifo(cur3, memo)
            spko_rec.append(spko)
            memo_rec.append(memo)
        
        return torch.stack(spko_rec, dim=0), torch.stack(memo_rec, dim=0)

