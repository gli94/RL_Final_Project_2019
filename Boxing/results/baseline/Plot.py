import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time
import matplotlib.pyplot as plt


R = np.loadtxt('average_return_dqn.txt')
L = np.loadtxt('average_return_a2c.txt')
plt.plot(np.arange(0, 588*100, 100), R, label='DQN')
plt.plot(np.arange(0, 588*100, 100), L, label='A2C')
plt.ylabel('Return')
plt.xlabel('Number of time steps')
plt.title('Learning Curve for CartPole-v0')
plt.legend()
plt.show()

