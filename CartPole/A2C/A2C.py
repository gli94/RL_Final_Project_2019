
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from multiprocessing_env import SubprocVecEnv
# Different from our DQN implementation which has been implemented from scratch,
# This code is based on https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch \
# to have the same parallel environments setting and framework

num_envs = 4
env_name = "CartPole-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk


envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def epsilon_greedy(model, state, num_actions, epsilon):
    if np.random.rand() > epsilon:
        _, action_value = model(state)
        v, a = action_value.max(1)
        a = a.item()
    else:
        a = np.random.randint(0, num_actions)
    return a


state_dim  = envs.observation_space.shape[0]
num_actions = envs.action_space.n

#Hyper params:
ALPHA = 1e-3
num_steps = 5

model = ActorCritic(state_dim, num_actions).to(device)
optimizer = optim.Adam(model.parameters())


max_frames   = 58800
frame_idx    = 0
test_rewards = []


state = envs.reset()

actor_losses = []
critic_losses = []
entropies = []

while frame_idx < max_frames:

    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            print(test_rewards[-1])
            
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)
    
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    # print('actor_loss:', actor_loss.item(), 'critic_loss:', critic_loss.item(), 'entropy:', entropy.item())
    actor_losses.append(actor_loss.item())
    # print('len:', len(actor_losses))
    critic_losses.append(critic_loss.item())
    entropies.append(entropy.item())

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

np.savetxt('actor_losses.out', actor_losses)
np.savetxt('critic_losses.out', critic_losses)
np.savetxt('entropies', entropies)
np.savetxt('rewards', test_rewards)

