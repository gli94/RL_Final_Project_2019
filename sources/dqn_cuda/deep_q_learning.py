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
from dqn import batch_wrapper, Phi

#import sys
#sys.path.append('../sources')

from replay_buffer import replay_buffer
from preprocessing import phi
import matplotlib.pyplot as plt
from preprocessing import phi
import matplotlib.pyplot as plt
from preprocessing import phi


# env = gym.make('CartPole-v0')
env = gym.make('Boxing-v0')

# Hyper Parameters
num_episode = 1000
BATCH_SIZE = 32
CAPACITY_SIZE = 10000
GAMMA = 0.99
ALPHA = 0.01
C = 500
N_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
HEIGHT = 28
WIDTH = 28

USE_GPU = True

return_per_episode = np.zeros(num_episode)

# Initialize the pre-processing function phi
# phi = Phi()

# Initialize experience replay buffer
buffer = replay_buffer(CAPACITY_SIZE)

# Initialize the targetNet and evalNet
# state_dim = (84, 84, 4)
# num_action = 18
        
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
        

# Initialize the behavior policy
# pi = epsilon_greedy(Q)

for episode in range(0, num_episode):
    start_time = time.time()
    x = env.reset()  # first frame
    # img = plt.imshow(env.render(mode='rgb_array'))  # only call this once

    s = [x]          # Initialize the sequence

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
        #print("\rPhi takes: %s seconds " % (time.time() - start_time))
        #env.render()

        a = Q.epsilon_greedy(p)
        x, r, done, _ = env.step(a)

        # TODO: reward clipping
        G += r
        s.append(a)      # can't quite get why a is stored into the sequence
        s.append(x)      # get s_{t+1}

        #start_time = time.time()
        p_next = phi(s, 4, HEIGHT, WIDTH)
        #print("\rPhi takes: %s seconds " % (time.time() - start_time))

        buffer.store(p, a, r, p_next, done)
        transBatch = buffer.sample(BATCH_SIZE)     # get a np.array
        phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch = batch_wrapper(transBatch)  # retrieve tensor batch
        
        phiBatch = phiBatch.to(device)
        actionBatch = actionBatch.to(device)
        rewardBatch = rewardBatch.to(device)
        phiNextBatch = phiNextBatch.to(device)

        # Q_value update: if next phi terminates, target is reward; else is reward + gamma * max(Q(phi_next, a'))
        nonFinalMask = torch.tensor(tuple(map(lambda m: m is not True, doneBatch)), dtype=torch.bool).to(device) # bool tensor [N]
        nextQ_Batch = torch.zeros(phiBatch.size()[0]).to(device)
        nextQ_Batch = torch.unsqueeze(nextQ_Batch, 1)      # nextQ_Batch shape(N, 1)

        nnInput = phiNextBatch[nonFinalMask].float()       # shape[N, 1], select non-terminal next state phi
        #start_time = time.time()
        nnOutput = Q.targetNet(nnInput)                    # size[N, 1]
        #print("\rTarget net inference takes: %s seconds " % (time.time() - start_time))

        nextQ_max = nnOutput.max(1)[0].detach()
        nextQ_max = torch.unsqueeze(nextQ_max, 1)                  # size[N, 1]

        nextQ_Batch[nonFinalMask] = nextQ_max

        targetBatch = (nextQ_Batch * GAMMA) + rewardBatch     # size[N, 1]
        
        #start_time = time.time()
        Q.update(phiBatch, actionBatch, targetBatch)
        #print("\rEval net train takes: %s seconds " % (time.time() - start_time))

        #############################
        # shape indicator
        # shape1 = phiBatch.size()
        # shape2 = actionBatch.size()
        # shape3 = targetBatch.size()
        # shape4 = rewardBatch.size()
        # shape5 = nextQ_Batch.size()
        #############################

        print("\rt {}".format(t), end="")

        if done:
            break
            
    print("\n")
    print('episode:', episode, 'return', G)
    return_per_episode[episode] = G
    print("One episode takes: %s seconds " % (time.time() - start_time))

PATH = './dqn_eval_net.pth'
torch.save(Q.evalNet.state_dict(), PATH)

with open('return_per_episode.txt', 'w') as f:
    for item in return_per_episode:
        f.write("%s\n" % item)

