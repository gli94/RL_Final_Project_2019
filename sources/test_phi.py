#import gym
#env = gym.make('Boxing-v0')
#env.reset()
#for _ in range(10000):
#	env.render()
#	env.step(env.action_space.sample())
#env.close()

from sources.preprocessing import phi
import numpy as np
import torch

a = np.ones((210, 160, 3))
seq = [a,0,2*a,0,3*a,0,4*a,0,5*a]

b = phi(seq, 4, 105, 80)

print(b.shape)
print(b)


