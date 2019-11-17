# This is a simplified version of deep Q learning algorithm from the Nature paper "Human-level control through deep
# reinforcement learning"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from dqn import DQN

env = gym.make('CartPole-v0')

# Initialize the behavior policy
pi = eplsilon_greedy()

# Initialize experience replay buffer

# Initialize the targetNet and evalNet
    Q = DQN()

    for episode in range(1, num_episode):
        x = env.reset()
        s = [x]

        for t in range(1, T)
            a = pi(phi(s), DQN.evalNet)
            x, r, done, _ = env.step(a)
            phi_t = phi(s)
            s.append(a, x)
            phi_t+1 = phi(s)

            buffer.store(phi_t, a, r, phi_t+1)
            transitionBatch = buffer.sample

            y = []
            for state in stateBatch:
                if(transition.phi_j+1 == termination):
                    y_j = transition.r_j
                else:
                    y_j = r_j + gamma * np.argmax(Q.targetNet(phi_j+1))

                y.add(y_j)

            Q.update(stateBatch, y)