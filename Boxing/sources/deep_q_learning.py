# This is a simplified version of deep Q learning algorithm from the Nature paper "Human-level control through deep
# reinforcement learning"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import time

from dqn import DQN
from dqn import batch_wrapper, Phi, compute_nextQ_batch, debug

from replay_buffer import replay_buffer
from preprocessing import phi
import matplotlib.pyplot as plt

env = gym.make('Boxing-ram-v0')

# Hyper Parameters
DEBUG = False
TOTAL_NUM_STEP = 50000000
num_episode = 10000
MODEL_SAVE_INTERVAL = 1000
BATCH_SIZE = 32
INITIAL_SIZE = 50000
CAPACITY_SIZE = 1000000
GAMMA = 0.99
ALPHA = 0.0001
C = 10000
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
        
return_per_episode = np.zeros(num_episode)
loss_per_step = []
total_t = 0

if DEBUG:
    EPSILON = []


for episode in range(0, num_episode):
    x = env.reset()
    s = [x]
    G = 0

    for t in count():
        # don't do anything in first 4 frames
        if t < 4:
            a = np.random.randint(0, N_ACTIONS)
            x, r, done, _ = env.step(a)
            s.append(x)
            continue
        p = Phi(s)
    
        a, epsilon = Q.epsilon_greedy(phi=p, epsilon_start=1, epsilon_end=0.1, decay_steps=1000000, total_t=total_t)

        if DEBUG:
            EPSILON.append(epsilon)

        x, r, done, _ = env.step(a)

        # clip rewards between -1 and 1
        r = max(-1.0, min(r, 1.0))

        G += r
        s.append(x)      # get s_{t+1}

        p_next = Phi(s)
        buffer.store(p, a, r, p_next, done)

        # Update evalNet every NUM_SKIP frames
        transBatch = buffer.sample(BATCH_SIZE)                                                   # get a np.array
        phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch,device)  # tensor batches

        # Q_value update: if next phi terminates, target is reward; else is reward + gamma * max(Q(phi_next, a'))
        nextQ_Batch = compute_nextQ_batch(Q=Q, phiBatch=phiBatch, phiNextBatch=phiNextBatch, doneBatch=doneBatch,
                                              device=device)

        targetBatch = (nextQ_Batch * GAMMA) + rewardBatch
        loss = Q.update(phiBatch, actionBatch, targetBatch)
        loss_per_step.append(loss)

        if DEBUG:
            debug(phiBatch, actionBatch, targetBatch, rewardBatch, nextQ_Batch, s, p)

        total_t += 1
        if total_t > TOTAL_NUM_STEP:
            break

        if done:
            break

    if DEBUG:
        break
            
    if episode % 1 == 0:
        print('episode:', episode, 'return', G)
    return_per_episode[episode] = G

    if episode % MODEL_SAVE_INTERVAL == 0:
        PATH = './dqn_eval_net_' + str(episode) + '.pth'
        torch.save(Q.evalNet.state_dict(), PATH)

PATH = './dqn_eval_net.pth'
torch.save(Q.evalNet.state_dict(), PATH)

with open('return_per_episode.txt', 'w') as f:
    for item in return_per_episode:
        f.write("%s\n" % item)

with open('loss_per_step.txt', 'w') as f:
    for item in loss_per_step:
        f.write("%s\n" % item)

if DEBUG:
    plt.plot(EPSILON)
    plt.ylabel('EPSILON')
    plt.show()

