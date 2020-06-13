from __future__ import absolute_import

from networks.utils import Flatten, PrintLayer, Branch, calculate_entropy
from ..branchynet.net import BranchyNet
import os, glob, sys, random, h5py
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import cv2



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
        nn.Conv2d(32, 64,  5, padding=2, stride=1),
        Branch(branch1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75),
        nn.Conv2d(64, 96,  3, padding=1, stride=1),
        Branch(branch2),
        nn.ReLU(),
        nn.Conv2d(96, 96,  3, padding=1, stride=1),
        Branch(branch3),
        nn.ReLU(),
        nn.Conv2d(96, 64,  3, padding=1, stride=1),
        Branch(branch4),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5, True),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5, True),
        nn.Linear(128, 10)        
    ]
    return network





class AlexNet(nn.Module):
    """
    Generate the architecture of AlexNet with four side branches or four early exits

    Attributes:
    ----------

    branch1, branch2, branch3, branch4: Branch(nn.Module)
        The ordered side branches inserted in main branch, which is AlexNet

    Methods:
    ----------
    forward: is executed to run the network 

    forward_train: is executed to train the network
    """


    def __init__(self, branch1, branch2, branch3, branch4, n_classes=1000):
        super(AlexNet, self).__init__()
        #Input Size: 224x224x3
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)

        self.lrn  = nn.LocalResponseNorm(3, alpha=5e-05, beta=0.75)

        self.max_pooling = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(32, 64,  5, padding=2, stride=1)

        self.exit1 = Branch(branch1)

        self.conv3 = nn.Conv2d(64, 96,  3, padding=1, stride=1)

        self.exit2 = Branch(branch2)

        self.conv4 = nn.Conv2d(96, 96,  3, padding=1, stride=1)

        self.exit3 = Branch(branch3)

        self.conv5 = nn.Conv2d(96, 64,  3, padding=1, stride=1)

        self.exit4 = Branch(branch4)

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=(1024), out_features=256),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256), out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=128, out_features=10))

    def forward_train(self, x):
        """
        Execute the forward propagation in neural networks in training process

        Parameters:
        x: Pytorch Tensor
        x.shape -> [3, 224, 224]
        """

        #Input Size: 224x224x3
        x = F.relu(self.conv_block1(x))

        x = self.max_pooling(x)

        x = self.lrn(x)

        x = self.conv2(x)

        x_exit1 = self.exit1(x)

        x = self.max_pooling(F.relu(x))

        x = self.lrn(x)

        x = self.conv3(x)

        x_exit2 = self.exit2(x)

        x = F.relu(x)

        x = self.conv4(x)

        x_exit3 = self.exit3(x)

        x = self.conv5(x)

        x_exit4 = self.exit4(x)

        x = F.relu(x)

        x = self.max_pooling(x)
    
        x_final = self.classifier(x)

        return x_final
        
    
    
    def forward(self, x, threshold_list=[0,0,0], mode_train=True):
        """
        It is executed to train the network
        Parameters:
        -----------
        x: Pytorch Tensor; x.shape -> [3, 224, 224]
        x is a sample to be classified

        threshold_list: List
        Define the entropy threshold to determinar if a sample classification 
        is confident enough to exit at a given side branch

        mode_train: bool
        Identify if training or inference process

        return: a Pytorch Tensor

        """

        if(mode_train):
            return self.forward_train(x)
        else:
            return self.forward_test_batch(x, threshold_list)


def get_network(percentTrainKeeps=1):
    """
    Built a BranchyNet, where the main branch is a AlexNet and four side branches are inserted 
    in main branch.
    """

    branch1 = norm() + conv(64) + conv(32) + cap(512)
    branch2 = norm() + conv(96) + cap(128)
    branch3 = conv(96) + cap(128)
    branch4 = conv(32) + cap(512)

    network = gen_2b(branch1, branch2, branch3, branch4)
    net = BranchyNet(network)


    return net
