# This is a simplified version of deep Q learning algorithm from the Nature paper "Human-level control through deep
# reinforcement learning"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
from src.dqn import DQN
from src.dqn import batch_wrapper, Phi
from sources.replay_buffer import replay_buffer
from sources.preprocessing import phi

env = gym.make('CartPole-v0')

# Hyper Parameters
num_episode = 400
BATCH_SIZE = 32
CAPACITY_SIZE = 10000
GAMMA = 0.99
ALPHA = 0.01
C = 4
N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]


# Initialize the pre-processing function phi
# phi = Phi()

# Initialize experience replay buffer
buffer = replay_buffer(CAPACITY_SIZE)

# Initialize the targetNet and evalNet
# state_dim = (84, 84, 4)
# num_action = 18

Q = DQN(state_dim=STATE_DIM,
        num_action=N_ACTIONS,
        alpha=ALPHA,
        C=C)

# Initialize the behavior policy
# pi = epsilon_greedy(Q)

for episode in range(0, num_episode):
    x = env.reset()  # first frame
    s = [x]          # Initialize the sequence

    for t in count():
        p = Phi(s)   # get phi_t
        a = Q.epsilon_greedy(p)

        x, r, done, _ = env.step(a)
        # s.append(a) # can't quite get why a is stored into the sequence
        s.append(x)  # get s_{t+1}
        p_next = Phi(s)  # get phi_{t+1}

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
