import numpy as np
import os, cv2, glob, sys, random, h5py
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import types, time
#from net import *


def train(branchyNet, x_train,y_train, batchsize=10000, num_epoch=20, main=False):
	datasize = x_train.shape[0]

	losses_list = []
	acc_list = []
	epoch_list = range(num_epoch)

	for epoch in epoch_list:
		print("Epoch: %s"%epoch)

		indexes = np.random.permutation(datasize)
		sum_loss = 0
		num = 0
		avglosses = []
		avgaccuracies = []
		diff_time = []

		for i in range(0, datasize, batchsize):
			input_data = Variable(x_train[indexes[i : i + batchsize]].to(self.device))
			label_data = Variable(y_train[indexes[i : i + batchsize]].to(self.device))

			start_time = time.time()
			if main:
				losses, acc = branchyNet.train_main(input_data, label_data)
			else:
				losses, acc = branchyNet.train_branch(input_data, label_data)

			print(acc)
			end_time = time.time()
			diff = end_time - start_time


			diff_time
			avglosses.append(losses)
			avgaccuracies.append(accuracies)


		avgloss = np.mean(np.array(avglosses),0)
		avgaccuracy = np.mean(np.array(avgaccuracies),0)

		losses_list.append(avgloss)
		acc_list.append(avgaccuracy)


		print("LOsses: %s"%(avg_loss))
		print("Acc: %s"%(avg_acc))

	return losses_list, acc_list
