import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time
import matplotlib.pyplot as plt


R = np.loadtxt('return_per_episode_baseline.txt')
L1 = np.loadtxt('return_per_episode_decay_rate_30000.txt')
L2 = np.loadtxt('return_per_episode_decay_rate_100000.txt')
plt.plot(R[0:1999], label='Decay Steps = 1M')
plt.plot(L1, label='Decay Steps = 30K')
plt.plot(L2, label='Decay Steps = 100K')
plt.ylabel('Clipped Return')
plt.xlabel('Number of Episodes')
plt.title('Learning Curve for Boxing-ram-v0')
plt.legend()
plt.show()

