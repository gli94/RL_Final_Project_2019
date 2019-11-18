# This is a simplified version of deep Q learning algorithm from the Nature paper "Human-level control through deep
# reinforcement learning"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
from src.dqn import DQN
from src.dqn import batch_wrapper
from sources.replay_buffer import replay_buffer

env = gym.make('CartPole-v0')

# Hyper Parameters
num_episode = 400
BATCH_SIZE = 32
GAMMA = 0.99
ALPHA = 0.01
C = 4

# Initialize the pre-processing function phi
phi = Phi()

# Initialize experience replay buffer
buffer = replay_buffer()

# Initialize the targetNet and evalNet
state_dim = (84, 84, 4)
num_action = 18

Q = DQN(state_dim=state_dim,
        num_action=num_action,
        alpha=ALPHA,
        C=C)

# Initialize the behavior policy
pi = epsilon_greedy(Q)

for episode in range(0, num_episode):
    x = env.reset()  # first frame
    s = [x]          # Initialize the sequence

    for t in count():
        p = phi(s)   # get phi_t
        a = pi(phi(s))

        x, r, done, _ = env.step(a)
        s.append(a)
        s.append(x)  # get s_{t+1}
        p_next = phi(s)  # get phi_{t+1}

        buffer.store(p, a, r, p_next, done)
        transBatch = buffer.sample(BATCH_SIZE)  # get a np.array
        phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch)  # retrieve tensor batch
        nonFinalMask = torch.tensor(tuple(map(lambda m: m is not True, doneBatch)), dtype=torch.uint8)

        # Q_value update: if next phi terminates, target is reward; else is reward + gamma * max(Q(phi_next, a'))
        nextQ_Batch = torch.zeros(BATCH_SIZE)
        nextQ_Batch[nonFinalMask] = Q.targetNet(phiNextBatch(nonFinalMask)).max(1)[0].detach()
        targetBatch = (nextQ_Batch * GAMMA) + rewardBatch

        # update evalNet every time; update targetNet every C time
        Q.update(phiBatch, targetBatch)

        if done:
            break