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

transBatch = [([1, 1, 1, 1], 2, 3, [1, 2, 2, 2], False),
              ([1, 1, 2, 1], 4, 5, [1, 3, 3, 3], True)
              ]
phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch)
print("phiBatch=", phiBatch)
print("phiBatch size:", phiBatch.size())
print("actionBatch=", actionBatch)
print("actionBatch size:", actionBatch.size())
print("rewardBatch", rewardBatch)
print("phiNextBatch=", phiNextBatch)
print("doneBatch=", doneBatch)

# nonFinalMask = torch.tensor(tuple(map(lambda m: m is not True, doneBatch)), dtype=torch.uint8)
d = map(lambda m: m is not True, doneBatch)

nonFinalMask = torch.tensor(tuple(map(lambda m: m is not True, doneBatch)), dtype=torch.bool)
nextQ_Batch = torch.zeros(2)
input = phiNextBatch[nonFinalMask].float()

nnOuput = torch.FloatTensor([[1, 2]])
index = torch.LongTensor([[1]])
q_values = nnOuput.gather(1, index)
print(q_values)
d = torch.rand(1, 2)
c = d.max(1)
print('d=', d)
print('c=', c)

