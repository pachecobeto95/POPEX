import numpy as np
import os, glob, sys, random, h5py
#import cv2
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import types, time
from networks.utils import Flatten, PrintLayer, Branch, calculate_entropy
import pandas as pd

class DNN(nn.Module):
	def __init__(self, layer_list, partitioning_layer):
		super(DNN, self).__init__()
		self.layer_list = layer_list
		self.partitioning_layer = partitioning_layer
		self.layers = nn.ModuleList(self.layer_list)

	def forward(self, inputs, n_model):
		cont = 5
		dict_time = {}
		dict_runtime = {}
		totaltime = 0
		j = 0
		#dict_size = {}
		if(self.partitioning_layer == "input"):
			return inputs, None

		for i, layer in enumerate(self.layer_list):
			start_time = time.time()
			inputs = layer(inputs)
			endtime = time.time()
			totaltime += endtime - start_time

			if(isinstance(layer, nn.Conv2d)):
				j+=1
				dict_runtime["l%s"%(j)] = totaltime
				#dict_size["l%s"%(j)] = inputs.cpu().detach().numpy().nbytes
				totatime = 0 
				cont_vertex+=1
				if(int(self.partitioning_layer[-1]) == cont_vertex):
					print("entrou")
					break

			elif(isinstance(layer, Branch)):
				dict_runtime["b%s"%(n_model)] = totaltime  

		return inputs, dict_runtime
		#return inputs, dict_time, dict_size



class DNN2(nn.Module):
  def __init__(self, main_branch, side_branch):
    super(DNN2, self).__init__()
    self.main_branch = main_branch
    self.main_branch.extend([side_branch])
  
  def forward(self, inputs, n_model):
    dict_runtime = {}
    #dict_size = {}
    totaltime = 0
    j = 0
    #k=0
    for i, layer in enumerate(self.main_branch):
      start_time = time.time()
      inputs = layer(inputs)
      endtime = time.time()
      totaltime += endtime - start_time

      if(isinstance(layer, nn.Conv2d) and (i>0)):
        j+=1
        dict_runtime["l%s"%(j)] = totaltime
        #dict_size["l%s"%(j)] = inputs.cpu().detach().numpy().nbytes
        totatime = 0 
      elif(isinstance(layer, Branch)):
        dict_runtime["b%s"%(n_model)] = totaltime  
    
    return inputs, dict_runtime
    #return inputs, dict_runtime, dict_size



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






	def __init__(self, network, partitioning_layer, thresholdExits=None, percentTestExits=.9, percentTrainKeeps=1., 
		lr=0.1, momentum=0.9, weight_decay=0.0001, alpha=0.001, opt="MomentumSGD", joint=True, 
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
		self.thresholdExits = thresholdExits

		self.main_list = []
		self.models = []
		self.error = nn.CrossEntropyLoss()

		starti = 0
		curri = 0

		for layer in network:
			side_branch = []
			if not isinstance(layer,Branch):
				curri += 1
				self.main_list.append(layer)

			else:

				#branch_model = DNN(self.main_list[starti:curri]+[layer])
				#self.models.append(branch_model)
				aux_starti = starti
				starti = curri
				endi = curri

				for prevlink in self.main_list:
					side_branch.append(prevlink)
		 		
				side_branch.append(layer)
					
				self.models.append(side_branch)



		self.main = DNN(self.main_list, partitioning_layer)
		self.models = [DNN(models, partitioning_layer) for models in self.models]

		if self.opt == 'MomentumSGD':
			self.optimizer_main = optim.SGD(self.main.parameters(), lr=self.lr, momentum=self.momentum, 
				weight_decay=self.weight_decay)

		else:
			self.optimizer_main = optim.Adam(self.main.parameters(), lr=self.lr, 
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)


		#for model in self.models:
		#	if self.opt == 'MomentumSGD':
		#		optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, 
		#		weight_decay=self.weight_decay)
		#	else:
		#		optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)

		#	self.optimizers.append(optimizer)            

		self.percentTrainKeeps = percentTrainKeeps
		self.percentTestExits = percentTestExits
		self.thresholdExits = thresholdExits
		self.clearLearnedExitsThresholds()
		self.verbose = verbose
		self.gpu = False
		self.xp = np

	def branch_models(self):
		return self.models 

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
		for i, model in enumerate(self.models,1):
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
    "best_loss":best_loss,
    "best_acc":best_acc}

		torch.save(main_state_dict, path)


	def save_branches(self, best_loss, best_acc,path):
		dict_models_branches = {}
		for i, model in enumerate(self.models, 1):
			dict_models_branches["model_branch_%s_state_dict"%(i)] = model.state_dict()
		
		dict_models_branches["optim_branch_%s_state_dict"] = self.optimizers.state_dict()
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

		return lossesdata, acc.cpu().numpy()

	def train_branch(self, x, t=None):
		numexits = []
		losses = []
		accuracies = []

		self.models, optimizer = self.initiate_branch()

		num_models = len(self.models)
		numsamples = x.shape[0]
		#print(self.main.layers[0].weight)

		for i, model in enumerate(self.models):
			y_pred = model(x)
			#print(y_pred)
			loss = self.error(y_pred, t.long()).cuda()
			losses.append(loss)
			accuracies.append(self.calculate_accuracy(y_pred, t))

			
			#optimizer.zero_grad()
			#loss.backward()
			#optimizer.step()
		#print(losses)
		optimizer.zero_grad()
		total_loss = sum(losses)	
		total_loss.backward()
		optimizer.step()

		#for loss in losses:
		#	optimizer.zero_grad()
		#	loss.backward() 
		#optimizer.step()



		lossesdata = [loss.item() for loss in losses]
		accuraciesdata = [accuracy for accuracy in accuracies]



		return lossesdata, accuraciesdata

	def initiate_branch(self):
		#self.models = []
		main_list = []
		curri = 0
		cont = 0
		offset = [0,0,0,0,3]

		#for param in self.main.parameters():
		#	param.requires_grad = False
		"""
		for i, layer in enumerate(self.network, 0):
			curri +=1
			#print(layer)
			#print("")
			if (isinstance(layer, Branch)):
				cont+=1

				branch_model = DNN2(self.main.layers[0:i-offset[cont]], layer)
				self.models.append(branch_model)
		"""		
		params = []
		for model in self.models:
			params += list(model.parameters()) 
		
		
		self.optimizers = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)
    
		return self.models, self.optimizers

	def test_main(self, x, t=None):
		start_time = time.time()
		h = self.main.test(x)
		endtime = time.time()
		inf_time = endtime - start_time

		accuracy = calculate_accuracy(h, t)
		if verbose:
			print("Accuracy: %s"%(accuracy))
	 
		return accuracydata, inf_time


	def test_branches_new(self, x, t=None):
		numexits = []
		numexits_keep = []
		accuracies = []

		remainingXVar = x
		remainingTVar = t

		keepingXVar = x
		keepingTVar = t		

		nummodels = len(self.models)
		numsamples = x.data.shape[0]
        
		totaltime = 0
		time_branches = []

		for i, model in enumerate(self.models):
			if (remainingXVar is None) or (remainingTVar is None):
				numexits.append(0)
				accuracies.append(0)
				time_branches.append(0)
				h_keep = model(keepingXVar)
				entropy_keep = calculate_entropy(h_keep)
				idx_keep = np.zeros(entropy_keep.shape[0],dtype=bool)
				idx_keep[entropy_keep < self.thresholdExits[i]] = True
				numexits_keep.append(sum(idx_keep))		
				continue
			

			h_keep = model(keepingXVar)
			entropy_keep = calculate_entropy(h_keep)
			idx_keep = np.zeros(entropy_keep.shape[0],dtype=bool)
			idx_keep[entropy_keep < self.thresholdExits[i]] = True
			numexits_keep.append(sum(idx_keep))



			start_time = time.time()
			h = model(remainingXVar)		
			endtime = time.time()
			totaltime += (endtime - start_time)

			time_branches.append(endtime - start_time)
			
			entropy = calculate_entropy(h)
			idx = np.zeros(entropy.shape[0],dtype=bool)
			idx[entropy < self.thresholdExits[i]] = True
			numexit = sum(idx)

			#if (i == nummodels-1):
			#	idx = np.ones(entropy.shape[0],dtype=bool)
			#	numexit = sum(idx)
		
			#else:
			#	if(self.thresholdExits is not None):
			#		idx[entropy < self.thresholdExits[i]] = True
			#		numexit = sum(idx)
	 
			total = entropy.shape[0]
			numkeep = total - numexit
			
			numexits.append(numexit)
	 
			x_data = remainingXVar
			t_data = remainingTVar
			if(numkeep > 0):
				remainingXVar = x_data[~idx]
				remainingTVar = t_data[~idx]				

			else:
				remainingXVar = None
				remainingTVar = None				
				
			if(numexit > 0):

				x_exit = x_data[idx]
				t_exit = t_data[idx]
				h_exit = model(x_exit)
		
				accuracy = float(self.calculate_accuracy(h_exit,t_exit).cpu().detach().numpy())
			
				accuracies.append(accuracy)

			else:
				accuracies.append(0)
				#acc_branch.append(0)		
		
		#acc_branch = np.array(acc_branch)
		accuracies = np.array(accuracies)	
		
		
		overall = 0
		for i, accuracy in enumerate(accuracies):
			overall += accuracy*numexits[i]

		
		overall /= numsamples
		totaltime /= numsamples
		time_branches = np.array(time_branches)/numsamples
		numexits = np.array(numexits)
		numexits_keep = np.array(numexits_keep)	
		ratio_exits = numexits/numsamples
		#accuracies = (accuracies*numexits)/numsamples


		#print("overall ahahaha %s, n_exits %s, accuracies %s"%(overall, numexits, accuracies))
		#print("acc_branch %s"%(acc_branch))
		#print(accuracies)
		#print((accuracies*numexits)/numsamples)	
		#print("numexits %s"%(numexits))
		#print("time_branches"%(time_branches))
		#print("totaltime %s"%(totaltime))

		return overall, accuracies, numexits, numexits_keep, ratio_exits, totaltime, time_branches


	def test_branches(self, x, t=None):

		remainingXVar = x
		remainingTVar = t 
		branches_time = []
		acc_list = []
		acc_list_threshold = []
            
		nummodels = len(self.models)
		numsamples = x.data.shape[0]

		dict_exits = {"branch_%s"%(i) :0 for i in range(1,nummodels+1)}

		#self.thresholdExits = np.ones(nummodels)

		for i, (model,thresholdExit)  in enumerate(zip(self.models, self.thresholdExits), 1):
			if remainingXVar is None or remainingTVar is None:
				numexits.append(0)
				acc_list.append(0)
				continue

			start_time = time.time()
			h = model(remainingXVar)
			endtime = time.time()
	 
			acc_unit = self.calculate_accuracy(h, t).cpu().numpy()
			acc_list.append(acc_unit)
	 
			branches_time.append(endtime-start_time)

			entropy = calculate_entropy(h)
			if(entropy <= thresholdExit):
				dict_exits["branch_%s"%(i)] += 1
		
		return acc_list, dict_exits, branches_time
				

	def initiate_branch2(self):
		self.models = []
		main_list = []
		curri = 0
		cont = 0
		offset = [0,0,0,0,3]

		#for param in self.main.parameters():
		#	param.requires_grad = False

		for i, layer in enumerate(self.network, 0):
			curri +=1
			#print(layer)
			#print("")
			if (isinstance(layer, Branch)):
				cont+=1

				branch_model = DNN2(self.main.layers[0:i-offset[cont]], layer)
				self.models.append(branch_model)
		
		params = []
		self.models.append(self.main)
		for model in self.models:
			params += list(model.parameters()) 

		#self.optimizers = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)
	
		#return self.models, self.optimizers
		return self.models


 