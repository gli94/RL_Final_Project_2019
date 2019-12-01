import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time
import matplotlib.pyplot as plt

from dqn import DQN
from dqn import batch_wrapper, Phi

R = np.loadtxt('return_per_episode.txt')
print(R)
plt.plot(R)
plt.ylabel('Return per step')
plt.show()

L = np.loadtxt('loss_per_step.txt')
print(L)
plt.plot(L)
plt.ylabel('loss_per_step')
plt.show()