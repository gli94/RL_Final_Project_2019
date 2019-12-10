# This is a simplified version of deep Q learning algorithm from the Nature paper "Human-level control through deep
# reinforcement learning"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time
import statistics

from dqn import DQN
from dqn import batch_wrapper, Phi, compute_nextQ_batch, debug

#import sys
#sys.path.append('../sources')

from replay_buffer import replay_buffer
from preprocessing import phi
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
# env = gym.make('Boxing-ram-v0')

# Hyper Parameters
DEBUG = False
TOTAL_NUM_STEP = 50000
num_episode = 10000
MODEL_SAVE_INTERVAL = 1000
BATCH_SIZE = 5
INITIAL_SIZE = 2
CAPACITY_SIZE = 200
GAMMA = 0.99
ALPHA = 1e-3
C = 2
NUM_SKIP = 4
N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
HEIGHT = 28
WIDTH = 28

USE_GPU = True

# Initialize experience replay buffer
buffer = replay_buffer(CAPACITY_SIZE)

if USE_GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    
print(device)

Q = DQN(state_dim=STATE_DIM,
        num_action=N_ACTIONS,
        alpha=ALPHA,
        C=C,
        learning_start=INITIAL_SIZE,
        learningfreq=NUM_SKIP,
        height=HEIGHT,
        width=WIDTH,
        device=device)


def test():
    x = env.reset()
    returns = []
    for episode in range(0, 10):
        done1 = False
        x = env.reset()
        G = 0
        s = [x]

        for t in count():

            p = Phi(s)
            action_value = Q.evalNet(torch.from_numpy(p).type(torch.FloatTensor).to(device))
            value, action = action_value.max(0)
            a = action.item()

            x, r, done1, _ = env.step(a)

            #env.render()

            G += r
            s.append(x)

            if done1:
                break

        returns.append(G)
    x1 = env.reset()
    print(statistics.mean(returns))
    return(statistics.mean(returns))

return_per_episode = np.zeros(num_episode)
loss_per_step = []
total_t = 0
average_return = []
for episode in range(0, num_episode):
    done = False
    x = env.reset()
    s = [x]
    G = 0

    for t in count():
        p = Phi(s)
        #env.render()


        a, epsilon = Q.epsilon_greedy(phi=p, epsilon_start=0.1, epsilon_end=0.1, decay_steps=1, total_t=total_t)

        x, r, done, _ = env.step(a)

        G += r

        s.append(x)      # get s_{t+1}

        p_next = Phi(s)
        buffer.store(p, a, r, p_next, done)

        # Update evalNet every NUM_SKIP frames
        transBatch = buffer.sample(BATCH_SIZE)                                                   # get a np.array
        phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch, device)  # tensor batches

        # Q_value update: if next phi terminates, target is reward; else is reward + gamma * max(Q(phi_next, a'))
        nextQ_Batch = compute_nextQ_batch(Q=Q, phiBatch=phiBatch, phiNextBatch=phiNextBatch, doneBatch=doneBatch,
                                          device=device)

        targetBatch = (nextQ_Batch * GAMMA) + rewardBatch

        loss = Q.update(phiBatch, actionBatch, targetBatch)

        loss_per_step.append(loss)

        total_t += 1

        if total_t % 100 == 0:
            # print(total_t)
            average_return.append(test())

        if total_t > TOTAL_NUM_STEP:
            break

        if done:
            break
np.savetxt('average_return.out', average_return)
