import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time
import matplotlib.pyplot as plt


R = np.loadtxt('return_per_episode_baseline.txt')
L1 = np.loadtxt('return_per_episode_learning_rate_00025.txt')
L2 = np.loadtxt('return_per_episode_learning_rate_001.txt')
plt.plot(R[0:1999], label='Learning Rate = 0.0001')
plt.plot(L1, label='Learning Rate = 0.00025')
plt.plot(L2, label='Learning Rate = 0.001')
plt.ylabel('Clipped Return')
plt.xlabel('Number of Episodes')
plt.title('Learning Curve for Boxing-ram-v0')
plt.legend()
plt.show()

