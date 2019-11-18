# classes: target network, online network, experience replay buffer, pre-processing function phi
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple

Transition = namedtuple('Transition', ('phi', 'action', 'reward', 'phi_next', 'done'))

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
    def __init__(self,
                 state_dim=(84, 84, 4),
                 num_action=18,
                 alpha=0.01,
                 C=4):
        self.targetNet = Net(state_dim, num_action)
        self.evalNet = Net(state_dim, num_action)

        self.learnCounter = 0
        self.C = C          # Every C steps we clone evalNet to be targetNet

        self.optimizer = torch.optim.Adam(self.evalNet.parameters(), lr=alpha)
        self.loss_func = nn.MSELoss()

    def eval(self, phi, action):
        """
        inp:
            phi: input of nn, shape of [BATCHSIZE, 84, 84, 4]
            action: sampled actions, shape of [BATCHSIZE, ]
        return:
            q_values: values of each (s, a), shape of [BATCHSIZE, ]
        """
        phi = torch.unsqueeze(torch.FloatTensor(phi), 0)
        q_values = self.evalNet(phi).gather(1, action)
        return q_values

    def target(self, phi, action):
        phi = torch.unsqueeze(torch.FloatTensor(phi), 0)
        q_values = self.targetNet(phi).gather(1, action)
        return q_values

    def update(self,
               phiBatch: torch.Tensor,
               actionBatch: torch.Tensor,
               targetBatch: torch.Tensor):
        if self.learnCounter == 0:
            self.targetNet.load_state_dict(self.evalNet.state_dict())
        self.learnCounter  = (self.learnCounter + 1) % self.C

        prediction = self.eval(phiBatch, actionBatch)
        loss = self.loss_func(prediction, targetBatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self,
                       phi,
                       epsilon=0.1):

        if np.random.rand() > epsilon:
            action_value = self.evalNet.forward(phi)
            action = torch.max(action_value, 1)[1].data.numpy()[0, 0]
        else:
            action = np.random.randint(0, num_action)
        return action




def batch_wrapper(transBatch: np.array
                  ):
    batch = Transition(*zip(*transBatch))
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    phiBatch = torch.cat(batch.phi)
    actionBatch = torch.cat(batch.action)
    rewardBatch = torch.cat(batch.reward)
    phiNextBatch = torch.cat(batch.phi_next)
    doneBatch = torch.cat(batch.done)

    return phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch






