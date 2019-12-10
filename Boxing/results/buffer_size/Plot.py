import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time
import matplotlib.pyplot as plt


R = np.loadtxt('return_per_episode_baseline.txt')
L1 = np.loadtxt('return_per_episode_buffer_100000.txt')
L2 = np.loadtxt('return_per_episode_buffer_500000.txt')
L = np.loadtxt('return_per_episode_wo_target.txt')
plt.plot(R[0:1999], label='DQN with target net')
#plt.plot(L1, label='Replay buffer size = 100K')
#plt.plot(L2, label='Replay buffer size = 500K')
plt.plot(L, label='DQN without target net')
plt.ylabel('Clipped Return')
plt.xlabel('Number of Episodes')
plt.title('Learning Curve for Boxing-ram-v0')
plt.legend()
plt.show()

