import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
from src.dqn import DQN
from src.dqn import batch_wrapper, Phi
from sources.replay_buffer import replay_buffer

env = gym.make('CartPole-v0')

N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
C = 4
ALPHA = 0.01

Q = DQN(state_dim=STATE_DIM,
        num_action=N_ACTIONS,
        alpha=ALPHA,
        C=C)

s = []
x = np.array([0.1, 0.1, 0.1, 0.1])
s.append(x)
# p = Phi(s)

p= torch.tensor([0.1, 0.1, 0.1, 0.1])
action_values = Q.evalNet(p)
value, action = action_values.max(0)
# action = torch.max(action_values, 1)[1].data.numpy()[0, 0]
print(action_values)

################################

transBatch = [([1, 1, 1, 1], 2, 3, [1, 2, 2, 2], True)]
phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch)
print("phiBatch=", phiBatch)
print("actionBatch=", actionBatch)
print("rewardBatch", rewardBatch)
print("phiNextBatch=", phiNextBatch)
print("doneBatch=", doneBatch)