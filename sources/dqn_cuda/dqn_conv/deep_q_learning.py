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

#import sys
#sys.path.append('../sources')

from replay_buffer import replay_buffer
from preprocessing import phi
import matplotlib.pyplot as plt

# env = gym.make('CartPole-v0')
env = gym.make('Boxing-v0')

# Hyper Parameters
DEBUG = False
TOTAL_NUM_STEP = 5000000
num_episode = 2000
BATCH_SIZE = 32
CAPACITY_SIZE = 50000
GAMMA = 0.99
ALPHA = 0.0001
C = 5000
NUM_SKIP = 4
N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
HEIGHT = 84
WIDTH = 84

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
        height=HEIGHT,
        width=WIDTH,
        device=device)
        
return_per_episode = np.zeros(num_episode)
loss_per_step = []
total_t = 0

if DEBUG:
    EPSILON = []


for episode in range(0, num_episode):
    start_time = time.time()
    x = env.reset()
    s = [x]
    G = 0

    for t in count():
        # don't do anything in first 4 frames
        if t < 4:
            a = np.random.randint(0, N_ACTIONS)
            x, r, done, _ = env.step(a)
            s.append(a)
            s.append(x)
            continue
        #start_time = time.time()
        p = phi(s, 4, HEIGHT, WIDTH)
        #p = Phi(s)
        #env.render()

        #policy_time = time.time()
        if t % NUM_SKIP == 0:
            a, epsilon = Q.epsilon_greedy(phi=p, epsilon_start=1, epsilon_end=0.1, decay_steps=100000, total_t=total_t)
        #print("\r---policy step time %s s ---" % (time.time() - policy_time))

        if DEBUG:
            EPSILON.append(epsilon)

        x, r, done, _ = env.step(a)

        # clip rewards between -1 and 1
        r = max(-1.0, min(r, 1.0))

        G += r
        s.append(a)
        s.append(x)      # get s_{t+1}

        p_next = phi(s, 4, HEIGHT, WIDTH)
        # p_next = Phi(s)
        buffer.store(p, a, r, p_next, done)

        # Update evalNet every NUM_SKIP frames
        if t % NUM_SKIP == 0:
            transBatch = buffer.sample(BATCH_SIZE)                                                   # get a np.array
            phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch,device)  # tensor batches

            # Q_value update: if next phi terminates, target is reward; else is reward + gamma * max(Q(phi_next, a'))
            nextQ_Batch = compute_nextQ_batch(Q=Q, phiBatch=phiBatch, phiNextBatch=phiNextBatch, doneBatch=doneBatch,
                                              device=device)

            targetBatch = (nextQ_Batch * GAMMA) + rewardBatch
            #train_time = time.time()
            loss = Q.update(phiBatch, actionBatch, targetBatch)
            #print("\r---train time %s seconds ---" % (time.time() - train_time))
            loss_per_step.append(loss)

            if DEBUG:
                debug(phiBatch, actionBatch, targetBatch, rewardBatch, nextQ_Batch, s, p)

        total_t += 1
        if total_t > TOTAL_NUM_STEP:
            break

        print("\rt {}".format(t), end="")

        if done:
            break
        #print("\r---1 step time %s ms ---" % (time.time() - start_time))

    if DEBUG:
        break
            
    #print("\n")
    if episode % 5 == 0:
        print('episode:', episode, 'return', G)
    return_per_episode[episode] = G
    print("One episode takes: %s seconds " % (time.time() - start_time))

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

