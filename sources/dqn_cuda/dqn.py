# classes: target network, online network, experience replay buffer, pre-processing function phi
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple

Transition = namedtuple('Transition', ('phi', 'action', 'reward', 'phi_next', 'done'))


class Net(nn.Module):
    def __init__(self,
                 state_dim=(84, 84, 4),
                 num_action=18):
        """
        state_dim: nn input dimension which is (84, 84, 4)
        num_action: number of possible actions, 4 - 18 for Atari
        """

        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, num_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self,
                x: torch.Tensor):
        """
        phi: nn input dimension, shape being [84, 84, 4]
        """
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.out(x)
        return action_values

# First Convolution Layer: 16 8x8 filters with stride 4
# ReLu
# Second Convolution Layer: 32 4x4 filters with stride 2
# ReLu
# Third FC Layer: 256 units
# ReLu
# Ouput Layer: output units = num_action
class ConvNet(nn.Module):
    def __init__(self, num_action, height, width):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # compute the size of input to the fully connected layer
        def conv2d_size_out(size, kernel_size=6, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_height = conv2d_size_out(conv2d_size_out(height, 6, 2), 4, 2)
        conv_width = conv2d_size_out(conv2d_size_out(width, 6, 2), 4, 2)

        self.fc1 = nn.Linear(32 * conv_height * conv_width, 256)
        self.output = nn.Linear(256, num_action)

        self.conv_height = conv_height
        self.conv_width = conv_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 32 * self.conv_height * self.conv_width)
        x = F.relu(self.fc1(x))
        x = self.output(x)

        return x


class DQN(object):
    def __init__(self,
                 state_dim=(84, 84, 4),
                 num_action=18,
                 alpha=0.01,
                 C=4,
                 height=84,
                 width=84):

        self.state_dim = state_dim
        self.num_action = num_action

        # self.targetNet = Net(state_dim, num_action)
        # self.evalNet = Net(state_dim, num_action)

        self.targetNet = ConvNet(num_action, height, width)
        self.evalNet = ConvNet(num_action, height, width)

        self.learnCounter = 0
        self.C = C          # Every C steps we clone evalNet to be targetNet

        self.optimizer = torch.optim.RMSprop(self.evalNet.parameters(), lr=alpha)
        self.loss_func = nn.MSELoss()

    def eval(self, phi, action):
        """
        inp:
            phi: input of nn, shape of [BATCHSIZE, 84, 84, 4]
            action: sampled actions, shape of [BATCHSIZE, ]
        return:
            q_values: values of each (s, a), shape of [BATCHSIZE, ]
        """
        # phi = torch.unsqueeze(torch.FloatTensor(phi), 0)
        # action = torch.unsqueeze(action, 0)
        nnOuput = self.evalNet(phi)
        q_values = self.evalNet(phi).gather(1, action)
        return q_values

    def target(self, phi, action):
        # phi = torch.unsqueeze(torch.FloatTensor(phi), 0)
        q_values = self.targetNet(phi).gather(1, action)
        return q_values

    def update(self,
               phiBatch: torch.Tensor,
               actionBatch: torch.Tensor,
               targetBatch: torch.Tensor):
        if self.learnCounter == 0:
            self.targetNet.load_state_dict(self.evalNet.state_dict())
        self.learnCounter  = (self.learnCounter + 1) % self.C

        prediction = self.eval(phiBatch, actionBatch)
        loss = self.loss_func(prediction, targetBatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self,
                       phi,
                       epsilon=0.1):
        # TODO: use a decaying epsilon
        phi = torch.from_numpy(phi).type(torch.FloatTensor)
        phi = phi.unsqueeze(0)

        if np.random.rand() > epsilon:
            action_value = self.evalNet(phi)
            value, action = action_value.max(1)
            action = action.item()
        else:
            action = np.random.randint(0, self.num_action)
        return action


def batch_wrapper(transBatch: np.array
                  ):
    """
    inp:
        transBatch: ndarrays of tuples, tuples being [phi, action, reward, phiNext, done]
        phi of size (BATCHSIZE, 4), dtype = int64
        action of size(BATCHSIZE), dtype = int64
        reward of size(BATCHSIZE), dtype = int64
        phiNext of size(BATCHSIZE, 4), dtype = int64
        done of size(BATCHSIZE), dtype = boolean
    return:
        phiBatch of size(BATCHSIZE, 4), dtype = float_tensor
        actionBatch of size(BATCHSIZE, 1), dtype = long_tensor
        rewardBatch of size(BATCHSIZE, 1), dtype = float_tensor
        phiNextBatch of size(BATCHSIZE, 4), dtype = float_tensor
        done of size (BATCHSIZE), a tuple of booleans
    """
    batch = Transition(*zip(*transBatch))
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    phiBatch = np.asarray(batch.phi)
    phiBatch = torch.from_numpy(phiBatch)
    phiBatch = phiBatch.float()

    actionBatch = np.asarray(batch.action)
    actionBatch = torch.from_numpy(actionBatch)
    actionBatch = torch.unsqueeze(actionBatch, 1)
    actionBatch = actionBatch.long()  # gather function takes in long tensor

    rewardBatch = np.asarray(batch.reward)
    rewardBatch = torch.from_numpy(rewardBatch)
    rewardBatch = torch.unsqueeze(rewardBatch, 1)
    rewardBatch = rewardBatch.float()

    phiNextBatch = np.asarray(batch.phi_next)
    phiNextBatch = torch.from_numpy(phiNextBatch)
    phiNextBatch = phiNextBatch.float()

    doneBatch = batch.done

    return phiBatch, actionBatch, rewardBatch, phiNextBatch, doneBatch


def Phi(s):
    # p = torch.from_numpy(s[-1]).type(torch.FloatTensor)
    # p = torch.tensor(s[-1])
    p = s[-1]
    return p






