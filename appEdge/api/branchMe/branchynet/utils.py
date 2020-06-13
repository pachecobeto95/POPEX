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
			input_data = Variable(torch.from_numpy(x_test[i : i + batchsize]).to("cuda:0"))
			label_data = Variable(torch.from_numpy(y_test[i : i + batchsize]).to("cuda:0"))

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



def test(branchyNet, x_test, y_test=, batchsize=1, main=False):
	datasize = x_test.shape[0]

	overall = 0.
	totaltime = 0.
	nsamples = 0
	num_exits = np.zeros(branchyNet.numexits()).astype(int)
	# finals = []
	accbreakdowns = np.zeros(branchyNet.numexits())
	for i in range(0, datasize, batchsize):
		input_data = Variable(torch.from_numpy(x_test[i : i + batchsize]).to("cuda:0"))
		label_data = Variable(torch.from_numpy(y_test[i : i + batchsize]).to("cuda:0"))

		start = time.time()
		if main:
			acc, diff = branchyNet.test_main(input_data, label_data)
		else:
			acc, accuracies, test_exits, diff = branchyNet.test_branch(input_data, label_data)
			for i, exits in enumerate(test_exits):
				num_exits[i] += exits
			for i in range(branchyNet.numexits()):
				accbreakdowns[i] += accuracies[i]*test_exits[i]

		end_time = time.time()    
		diff = end_time-start_time

		totaltime += diff
		overall += input_data.shape[0]*acc
		nsamples += input_data.shape[0]

	overall /= nsamples

	for i in range(branchyNet.numexits()):
		if num_exits[i] > 0:
			accbreakdowns[i]/=num_exits[i]
	#if len(finals) > 0:
	#	hh = Variable(np.vstack(finals),volatile=True)
	#	tt = Variable(y_test, volatile=True)
	#	overall = F.accuracy(hh,tt).data

	return overall, totaltime, num_exits, accbreakdowns




def get_inc_points(accs, diffs, ts, exits, inc_amt=-0.0005):
    idxs = np.argsort(diffs)
    accs = np.array(accs)
    diffs = np.array(diffs)
    inc_accs = [accs[idxs[0]]]
    inc_rts = [diffs[idxs[0]]]
    inc_ts = [ts[idxs[0]]]
    inc_exits = [exits[idxs[0]]]
    for i, idx in enumerate(idxs[1:]):
        #allow for small decrease
        if accs[idx] > inc_accs[-1]+inc_amt:
            inc_accs.append(accs[idx])
            inc_rts.append(diffs[idx])
            inc_ts.append(ts[idx])
            inc_exits.append(exits[idx])
    
    return inc_accs, inc_rts, inc_ts, inc_exits


def generate_thresholds(thresholds_base, num_exits):
  ts = list(product(*([thresholds_base]*(num_exits))))
  ts = [list(l) for l in ts]
  return ts


def eval_branches_thresholds(branchyNet, x_test, y_test, thresholds_base,dataset_size, batchsize=1, verbose=False):
  df = pd.DataFrame(columns = ["thresholds", "acc", "inference_time", "num_exit", "exit_ratio"])
  thresholds_list = generate_thresholds(thresholds_base, branchyNet.numexits())[::-1]

  accs = []
  diffs = []
  num_exits_list = []
  print(thresholds_list)
  cont = len(df)
  df["thresholds"] = thresholds_list  
  for thresholds in thresholds_list[cont:]:
    print(thresholds)
    branchyNet.thresholdExits = thresholds    
    acc, totaltime, num_exit, accbreakdowns = test(branchyNet,x_test,y_test, batchsize=batchsize)
    df.loc[cont, "acc"] = acc
    df.loc[cont, "inference_time"] = totaltime
    df.loc[cont, "num_exit"] = num_exit
    df.loc[cont, "exit_ratio"] = np.array(num_exit)/float(dataset_size)
    cont += 1
    df.to_csv("save_eval_thesholds_cifar_10.csv")

