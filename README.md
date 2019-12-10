# Course Project for Reinforcement Learning: Theory and Practice

This project implements Deep-Q-Network (DQN), and applies DQN to the environment Boxing-ram-v0 to train the agent to play Boxing game.

# Dependencies

1. python 3.7.2
2. torch
3. torchvision
4. gym
5. gym[atari]
6. numpy
7. matplotlib
8. scikit-image

# Run the code

The following commands assume you have set python 3 as default python.

## Train the agent with DQN

```
cd ./Boxing/sources
python deep_q_learning.py
```

## Check pre-trained results

First, enter `demo` directory under `Boxing`:

```
cd ./Boxing/demo
```

### Run the agent with model trained after 10000 episodes

```
python demo.py -trained
```

### Run the agent with model trained less than 10000 episodes

For instance, run the agent with model trained after 5000 episodes:

```
python demo.py -trained -incremental 5000
```

Available pre-trained models are trained after 1000, 2000, 3000, ..., 10000 steps.

### Run the agent with random policy

```
python demo.py -random
```

## Train the agent for CartPole with A2C

```
cd ./CartPole/A2C
python A2C.py
```

## Train the agent for CartPole with DQN

```
cd ./CartPole/DQN
python deep_q_learning.py
```


# Demo Video

If you are insterested, please checkout our [demo video](https://www.youtube.com/watch?v=oluukG7qS0E&feature=youtu.be) for more details.













