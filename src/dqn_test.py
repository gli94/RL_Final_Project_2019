import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
from src.dqn import DQN
from src.dqn import batch_wrapper, Phi
from sources.replay_buffer import replay_buffer
import matplotlib.pyplot as plt

Return = np.array([])
Return = np.append(Return, 4)
np.savetxt('Return.out', Return, delimiter=' ')
L = np.loadtxt('Loss.out')
print(L)
L = L[L < 1]
plt.plot(L)
plt.ylabel('loss')
plt.show()