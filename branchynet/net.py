import numpy as np
import os, cv2, glob, sys, random, h5py
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import types, time
from networks.utils import Flatten, PrintLayer, Branch, calculate_entropy


class DNN(nn.Module):
	"""
	A class used to represent a general chain-topology DNN.

	Parameters:
	--------------
	layer_list: List
	This list contains the layers of a given DNN. This class receives, then mount
	the chain-topology.

	Methods:

	forward(input: Tensor)
        Returns the DNN output 
	"""
	def __init__(self, layer_list):
		super(DNN, self).__init__()
		self.layer_list = layer_list
		self.layers = nn.ModuleList(self.layer_list)

	def forward(self, inputs):

		for i in range(self.layer_list):
			inputs = self.layers[i](inputs)

		return inputs



class BranchyNet:

	"""
	A class used to represent a BranchyNet: A chain-topology DNN with early exits inserted at
	the middle layers.


	Parameters:
	----------------
	network: DNN class, a nn.ModuleList
		the topology of the DNN

	thresholdExits: List 
		Entropy threshold list of side branches in BranchyNet, where
		thresholdExits[i] corresponds the entropy threshold of side branch i (default None).

	The following are the hyperparameters used to training the network. 
	lr: int
		the learning rate used in optimization algorithm: SGD (Gradient descent), MomentumSGD 
		or even Adam (default 0.1)

	momentum: int
		momentum is a parameter used in MomentumSGD optimization algorithm 

	weight_decay: int
		a regularization parameter


	opt: str
		indicates the optimization algortithm (default Adam)

	verbose: bool
		prints the losses and accuracies of each batch (default False) 

	self.optimizers = List
		stores the optimization object of each side branch

	self.optimizer_main = optim object
		stores the optimization object of main branch


	Methods:
	----------------
	numexits():
		Returns the number of early exits (side branches) in BranchyNet 

	training():
		Activates the training mode in main branch and side branches

	testing():
		 Activates the testing mode in main branch and side branches

	to_gpu()
		Processes in GPU

	to_cpu()
		Processes in CPU


	save_main(best_loss, best_acc, path)
		Saves the main branch model, best loss and the bess accuracy in a state dictionary

	save_branches(best_loss, best_acc, path)
		Saves the side branches model, best loss and the bess accuracy in a state dictionary	


	calculate_accuracy(fx, y):
		Calculares the ratio between the unmber of correct predictions and the total 
		of predictions. 

	train_main(x, t=None)
		Train the main branch model (t -> default None)


	initiate_branch():
		Tranfer the trained layers of main branch to create a list of models 
		with the side branches

	"""






	def __init__(self, network,thresholdExits=None, percentTestExits=.9, percentTrainKeeps=1., 
		lr=0.1, momentum=0.9, weight_decay=0.0001, alpha=0.001, opt="Adam", joint=True, 
		verbose=False):
		
		
		self.opt = opt
		self.lr = lr
		self.alpha = alpha
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.joint = joint
		self.forwardMain = None
		self.optimizers = []
		self.network = network

		self.main_list = []
		self.models = []
		self.error = nn.CrossEntropyLoss()

		starti = 0
		curri = 0

		for layer in network:
			curri += 1
			if not isinstance(layer,Branch):
				self.main_list.append(layer)

			else:
				branch_model = DNN(self.main_list[starti:curri]+[layer])
				self.models.append(branch_model)
				

		self.main = DNN(self.main_list)

		if self.opt == 'MomentumSGD':
			self.optimizer_main = optim.SGD(self.main.parameters(), lr=self.lr, momentum=self.momentum, 
				weight_decay=self.weight_decay)

		else:
			self.optimizer_main = optim.Adam(params, lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)


		for model in self.models:
			if self.opt == 'MomentumSGD':
				optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, 
				weight_decay=self.weight_decay)
			else:
				optimizer = optim.Adam(params, lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

			self.optimizers.append(optimizer)            

		self.percentTrainKeeps = percentTrainKeeps
		self.percentTestExits = percentTestExits
		self.thresholdExits = thresholdExits
		self.clearLearnedExitsThresholds()
		self.verbose = verbose
		self.gpu = False
		self.xp = np


	def getLearnedExitsThresholds(self):
		return self.learnedExitsThresholds/self.learnedExitsThresholdsCount

	def clearLearnedExitsThresholds(self):
		self.learnedExitsThresholds = np.zeros(len(self.models))
		self.learnedExitsThresholdsCount = np.zeros(len(self.models))

	def numexits(self):
		return len(self.models)

	def training(self):
		self.main.train()
		for model in self.models:
			model.train()

	def testing(self):
		self.main.eval()
		for model in self.models:
			model.eval()

	def to_gpu(self):
		self.gpu = True
		self.main = self.main.cuda()
		self.models = [model.cuda() for model in self.models]

	def to_cpu(self):
		self.gpu = True
		self.main = self.main.to("cpu")
		self.models = [model.to("cpu") for model in self.models]


	def save_main(self, best_loss, best_acc, path):
		main_state_dict = {
		"model_main_state_dict": self.main.state_dict(),
		"optimizer_main_state_dict": self.optimizer_main.state_dict(),
		"best_loss": best_loss,
		"best_acc":best_acc}

		torch.save(main_state_dict, path)


	def save_branches(self, best_loss, best_acc,path):
		dict_models_branches = {}
		for i, (model, optim) in enumerate(zip(self.models, self.optimizers), 1):
			dict_models_branches["model_branch_%s_state_dict"%(i)] = model.state_dict()
			dict_models_branches["optim_branch_%s_state_dict"%(i)] = optim.state_dict()

		
		dict_models_branches["best_loss"] = best_loss
		dict_models_branches["best_acc"] = best_acc
		torch.save(dict_models_branches, path)


	def save_branchyNet(self, path):
		dict_models_branches = {}
		dict_models_branches["model_main_state_dict"] = self.main.state_dict()
		for i, (model, optim) in enumerate(zip(self.models, self.optimizers), 1):
			dict_models_branches["model_branch_%s_state_dict"%(i)] = model.state_dict()
			dict_models_branches["optim_branch_%s_state_dict"%(i)] = optim.state_dict()

		torch.save(dict_models_branches, path)


	def calculate_accuracy(self, fx, y):  

		preds = fx.max(1, keepdim=True)[1].t()

		correct = preds.eq(y.view_as(preds)).sum()

		acc = correct.float()/preds.shape[1]
		
		return acc


	def train_main(self, x, t=None):
		y_pred = self.main(x)

		
		loss = self.error(y_pred, t.long()).float().cuda()
		acc = self.calculate_accuracy(y_pred, t)

		lossesdata = loss.item()

		self.optimizer_main.zero_grad()
		loss.backward()
		self.optimizer_main.step()
		if self.verbose:
			print("Losses for minibatch: %s"%(lossesdata))
			print("Acc for minibatch: %s"%(acc))

		return lossesdata, acc


	def initiate_branch(self):
		models = []
		main_list = []
		curri = 0
		cont = 0
		offset = [0,0,0,3]

		for param in self.main.parameters():
			param.requires_grad = False

		for i, layer in enumerate(self.network, 0):
			curri +=1
			if (isinstance(layer, Branch)):
				cont+=1
				print("entrou", i, cont)
				print(self.main.layers[0:i+offset[cont]])
				branch_model = DNN(list(self.main.layers[0:i-offset[cont]])+[layer])
				models.append(branch_model)

		return models




	def train_branch(self, x, t=None):
		numexits = []
		losses = []
		accuracies = []

		self.models = self.initiate_branch()


		num_models = len(self.models)
		numsamples = x.shape[0]

		for i, model in enumerate(self.models):
			y_pred = model(x)

			loss = self.error(y_pred, t.long()).cuda()
			losses.append(loss)
			accuracies.append(self.calculate_accuracy(y_pred, t))

			if (i==num_models-1):
				continue

			softmax = F.softmax(y_pred)
			entropy_value = entropy_gpu(softmax).get()
			if self.gpu:
				entropy_value = entropy_gpu(softmax).get()
			else:
				entropy_value = np.array([entropy(s) for s in softmax.data])    


			total = entropy_value.shape[0]
			idx = np.zeros(total,dtype=bool)
			if i == nummodels-1:
				idx = np.ones(entropy_value.shape[0],dtype=bool)

			else:
				if self.thresholdExits is not None:
					min_ent = 0
					if isinstance(self.thresholdExits,list):
						idx[entropy_value < min_ent+self.thresholdExits] = True
						numexit = sum(idx)
					else:
						idx[entropy_value < min_ent+self.thresholdExits] = True
						numexit = sum(idx)

				elif hasattr(self,'percentTrainExits') and self.percentTrainExits is not None:
					if isinstance(self.percentTrainExits,list):
						numexit = int((self.percentTrainExits[i])*numsamples)
					else:
						numexit = int(self.percentTrainExits*entropy_value.shape[0])

					#esorted = entropy_value.argsort()
					#idx[esorted[:numexit]] = True

				else:
					if isinstance(self.percentTrainKeeps,list):
						numkeep = (self.percentTrainKeeps[i])*numsamples
					else:
						numkeep = self.percentTrainKeeps*total

					numexit = int(total - numkeep)
					esorted = entropy_value.argsort()
					idx[esorted[:numexit]] = True



			numkeep = int(total - numexit)
			numexits.append(numexit)

		for i,(loss, optimizer) in enumerate(zip(losses,self.optimizers)):
			optimizer.zero_grad()
			loss.backward() 
			optimizer.step()



		lossesdata = [loss.item() for loss in losses]
		accuraciesdata = [accuracy for accuracy in accuracies]

		if self.verbose:
			print ("numexits",numexits)
			print ("losses",lossesdata)
			print ("accuracies",accuraciesdata)

		return lossesdata,accuraciesdata




 