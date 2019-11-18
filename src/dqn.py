# classes: target network, online network, experience replay buffer, pre-processing function phi
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

class Net(nn.module):
    def __init__(self,
                 state_dim=(84, 84, 4),
                 num_action=18):
        """
        state_dim: nn input dimension which is (84, 84, 4)
        num_action: number of possible actions, 4 - 18 for Atari
        """

        super(Net, self).__init__()
        self.fc1 = nn.linear(state_dim, 10)
        self.fc1.weight.data.normal(0, 0.1)
        self.out = nn.Linear(10, num_action)
        self.out.weight.data.normal(0, 0.1)

    def forward(self,
                phi: torch.Tensor):
        """
        phi: nn input dimension, shape being [84, 84, 4]
        """
        phi = self.fc1(phi)
        phi = F.relu(phi)
        action_values = self.out(phi)
        return action_values

class DQN(object):
    def __init__(self, state_dim, num_action, alpha, C):
        self.targetNet = Net(state_dim, num_action)
        self.evalNet = Net(state_dim, num_action)

        self.learnCounter = 0
        self.C = C          # Every C steps we clone evalNet to be targetNet

        self.optimizer = torch.optim.Adam(self.evalNet.parameters(), lr=alpha)
        self.loss_func = nn.MSELoss()

    def eval(self, phi):
        phi = torch.unsqueeze(torch.FloatTensor(phi), 0)
        return self.evalNet(phi)

    def target(self, phi):
        phi = torch.unsqueeze(torch.FloatTensor(phi), 0)
        return self.targetNet(phi)

    def update(self,
               phiBatch: torch.Tensor,
               targetBatch: torch.Tensor):
        if self.learnCounter == 0:
            self.targetNet.load_state_dict(self.evalNet.state_dict())
        self.learnCounter  = (self.learnCounter + 1) % self.C

        prediction = self.evalNet(phiBatch)
        loss = self.loss_func(prediction, targetBatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



