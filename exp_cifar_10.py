"""
Author: Roberto Pacheco
Date: 16/03/2020
Objective:  This script trains a B-AlexNet to clasify imagens in Cifar 10 dataset

BranchyNet Paper: https://arxiv.org/abs/1709.01686 
"""

import torch
import torch.nn as nn
import numpy as np, sys
from networks import alex_cifar10
from datasets import pcifar10
from branchynet import utils
#import the hyperparameters
import config


#Get the B-AlexNet architecture to classify images using Cifar10 dataset
branchyNet = alex_cifar10.get_network()

x_train, y_train, x_test, y_test = pcifar10.get_data(config.DATASET_PATH)

branchyNet.training()

# Train only the main branch of BranchyNet. 
main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=config.TRAIN_BATCHSIZE,
	num_epoch=config.TRAIN_MAIN_NUM_EPOCHS)

# Train the side branches of BranchyNet. 
main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, batchsize=config.TRAIN_BATCHSIZE,
	num_epoch=config.TRAIN_BRANCH_NUM_EPOCHS)
