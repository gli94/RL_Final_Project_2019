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

import sys

from replay_buffer import replay_buffer
from preprocessing import phi
import matplotlib.pyplot as plt

env = gym.make('Boxing-ram-v0')

# Hyper Parameters
TOTAL_NUM_STEP = 50000000
num_episode = 10000
num_episode_test = 100
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
        
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

if len(sys.argv) > 1 and sys.argv[1] == "-trained":

    if len(sys.argv) > 3 and sys.argv[2] == "-incremental":
        PATH = './model/dqn_eval_net_' + str(sys.argv[3]) + '.pth'
        print('Loading model trained after %s steps...' % sys.argv[3])
    else:
        PATH = './model/dqn_eval_net.pth'
        
    Q.evalNet.load_state_dict(torch.load(PATH, map_location=device))
    print("Pre-trained model loaded!")
    
    for episode in range(0, num_episode):
        x = env.reset()
        G = 0
        s = [x]

        for t in count():
            
            p = Phi(s)
            action_value = Q.evalNet(torch.from_numpy(p).type(torch.FloatTensor).to(device))
            value, action = action_value.max(0)
            a = action.item()
            
            
            x, r, done, _ = env.step(a)
            env.render()
            time.sleep(0.05)

            G += r
            s.append(x)
       
            if done:
                break
        
            
        if episode % 1 == 0:
            print('episode:', episode, 'return', G)
            
            
elif len(sys.argv) > 1 and sys.argv[1] == "-test":

    rand_flag = False
    
    if len(sys.argv) > 2 and sys.argv[2] == "-random":
        rand_flag = True
        print("Use random policy!")
    elif len(sys.argv) > 3 and sys.argv[2] == "-incremental":
            PATH = './model/dqn_eval_net_' + str(sys.argv[3]) + '.pth'
            print('Loading model trained after %s steps...' % sys.argv[3])
            Q.evalNet.load_state_dict(torch.load(PATH, map_location=device))
            print("Pre-trained model loaded!")
    else:
        PATH = './model/dqn_eval_net.pth'
        Q.evalNet.load_state_dict(torch.load(PATH, map_location=device))
        print("Pre-trained model loaded!")
    
    acc_returns = 0

    for episode in range(0, num_episode_test):
        x = env.reset()
        G = 0
        s = [x]

        for t in count():
               
            p = Phi(s)
            action_value = Q.evalNet(torch.from_numpy(p).type(torch.FloatTensor).unsqueeze(0).to(device))
            value, action = action_value.max(1)
            
            if rand_flag:
                a = np.random.randint(0, N_ACTIONS)
            else:
                a = action.item()
               
            x, r, done, _ = env.step(a)

            G += r
            s.append(x)
          
            if done:
                break
        
        acc_returns += G
        print("\rEpisode #: {}".format(episode), end="")
    
    print("")
    print("Average returns over %d episodes is: %f" % (num_episode_test, acc_returns / num_episode_test))
               
        


else:
    print("Use random policy!")

    for episode in range(0, num_episode):
        x = env.reset()
        G = 0

        for t in count():

            a = np.random.randint(0, N_ACTIONS)
   
            x, r, done, _ = env.step(a)
            env.render()
            time.sleep(0.05)

            G += r
       
            if done:
                break
        
            
        if episode % 1 == 0:
            print('episode:', episode, 'return', G)

   




