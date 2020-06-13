import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from .utils import Flatten, PrintLayer, Branch, calculate_entropy
from ..branchynet.net import BranchyNet

def norm():
    """
    A function used to built-in block of layers to norm the input
    """

    return [nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75)]

def pool():
    """
    A function used to apply pool operation
    """

    return [nn.MaxPool2d(kernel_size=3, stride=2)]



#Built-in block of convolutional layer to extract features
conv = lambda n: [nn.Conv2d(n, 32,  3, padding=1, stride=1), nn.ReLU()]

#Built-in block of fully-connected layer to classify the samples
cap =  lambda n: [nn.MaxPool2d(kernel_size=3, stride=2), Flatten(), nn.Linear(n, 10)]


class AlexNet_Conv(nn.Module):
    def __init__(self):
        super(AlexNet_Conv, self).__init__()
        #Input Layer: 224x224x3
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        #Output first convolutional layer: 55x55x96
        self.lrn  = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.)
        #Input second convolutional layer: 27x27x96
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        #Output second convolutional layer: 27x27x256
        #Input third convolutional layer: 13x13x256     
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        #Output third convolutional layer: 13x13x384
        #Input fourth convolutional layer: 13x13x384
        self.conv4 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
        #Output fourth convolutional layer:13x13x384
        self.conv5 = nn.Conv2d(192, 256, kernel_size=3, padding=1)

    def forward(self, x):
        #First Conv Layer
        x = F.relu(self.conv1(x))
        x = self.lrn(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        #Second Conv Layer
        x = F.relu(self.conv2(x))
        x = self.lrn(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        #Third, fourth and fifth Conv Layers 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x.view(-1, 5*5*256)
        return x


def gen_2b(branch1=None, branch2=None, branch3=None, branch4=None):
    """
    Generate the architecture of AlexNet with four side branches or four early exits

    branch1, branch2, branch3, branch4: Branch(nn.Module)
        The ordered side branches inserted in main branch, which is AlexNet
    """
    
    network = [
        nn.Conv2d(3, 32, 5, padding=2, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
    ]
    return network


def get_network(partitioning_layer, percentTrainKeeps=1):
    """
    Built a BranchyNet, where the main branch is a AlexNet and four side branches are inserted 
    in main branch.
    """

    branch1 = norm() + conv(64) + conv(32) + cap(512)
    branch2 = norm() + conv(96) + cap(128)
    branch3 = conv(96) + cap(128)
    branch4 = conv(32) + cap(512)

    network = gen_2b(branch1, branch2, branch3, branch4)
    net = BranchyNet(network, partitioning_layer)

    return net

