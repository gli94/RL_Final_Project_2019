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
<<<<<<< HEAD
<<<<<<< HEAD
from sources.preprocessing import phi
=======
import matplotlib.pyplot as plt
>>>>>>> cdcde89227cefbf02a91ee115489782569b3604e
=======

from sources.preprocessing import phi
>>>>>>> be7a7e61de86d28d02df87e99b501992d8450582


# env = gym.make('CartPole-v0')
env = gym.make('Boxing-v0')

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
    # img = plt.imshow(env.render(mode='rgb_array'))  # only call this once

    s = [x]          # Initialize the sequence

    G = 0
    for t in count():
        # p = Phi(s)       # get phi_t

        # don't do anything in first 4 frames
        if t < 5:
            a = np.random.randint(0, N_ACTIONS)
            x, r, done, _ = env.step(a)
            s.append(a)
            s.append(x)
            continue

        p = phi(s)
        # env.render()
        a = Q.epsilon_greedy(p)

        x, r, done, _ = env.step(a)
        G += r
        s.append(a) # can't quite get why a is stored into the sequence
        s.append(x)      # get s_{t+1}
        # p_next = Phi(s)  # get phi_{t+1}
        p_next = phi(s)

        buffer.store(p, a, r, p_next, done)
        transBatch = buffer.sample(BATCH_SIZE)     # get a np.array
        phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch)  # retrieve tensor batch
        nonFinalMask = torch.tensor(tuple(map(lambda m: m is not True, doneBatch)), dtype=torch.bool)

        # Q_value update: if next phi terminates, target is reward; else is reward + gamma * max(Q(phi_next, a'))
        nextQ_Batch = torch.zeros(phiBatch.size()[0])
        nextQ_Batch = torch.unsqueeze(nextQ_Batch, 1)      # nextQ_Batch shape(N, 1)

        nnInput = phiNextBatch[nonFinalMask].float()       # shape[N, 1]
        nnOutput = Q.targetNet(nnInput)                    # size[N, 1]

        nextQ_max = nnOutput.max(1)[0].detach()
        nextQ_max = torch.unsqueeze(nextQ_max, 1)                  # size[N, 1]

        nextQ_Batch[nonFinalMask] = nextQ_max

        targetBatch = (nextQ_Batch * GAMMA) + rewardBatch     # size[N, 1]

        Q.update(phiBatch.float(), actionBatch, targetBatch.float())
        # shape indicator
        # shape1 = phiBatch.size()
        # shape2 = actionBatch.size()
        # shape3 = targetBatch.size()
        # shape4 = rewardBatch.size()
        # shape5 = nextQ_Batch.size()
        print(t)

        if done:
            break
<<<<<<< HEAD
<<<<<<< HEAD
=======
    print('episode:', episode, 'return', G)
>>>>>>> cdcde89227cefbf02a91ee115489782569b3604e
=======

    print('episode:', episode, 'return', G)
>>>>>>> be7a7e61de86d28d02df87e99b501992d8450582
