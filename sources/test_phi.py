#import gym
#env = gym.make('Boxing-v0')
#env.reset()
#for _ in range(10000):
#	env.render()
#	env.step(env.action_space.sample())
#env.close()

from preprocessing import phi
import numpy as np
import torch
import gym

#a = np.ones((210, 160, 3))
#seq = [a,0,2*a,0,3*a,0,4*a,0,5*a]

#b = phi(seq, 4, 105, 80)

env = gym.make('Boxing-v0')
state = env.reset()
print(state.shape)

seq = [state]

print(state[:,:,2])

b = phi(seq,1)
print(b)
print(b.shape)

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    seq.append(action)
    state, r, done, info = env.step(action)
    seq.append(state)

print(len(seq))

b = phi(seq, 4, 28, 28)

print(b)
print(b.shape)
    


