import gym
env = gym.make('Boxing-v0')
env.reset()
for _ in range(10000):
	env.render()
	env.step(env.action_space.sample())
env.close()
