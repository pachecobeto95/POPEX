import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from nn.branches import branch1

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class PrintLayer(nn.Module):
	def __init__(self):
		super(PrintLayer, self).__init__()

	def forward(self, x):
	# Do your print / debug stuff here
		print(x.shape)
		return x

class Branch(nn.Module):
	def __init__(self, layer):
		super(Branch, self).__init__()
		if(layer is not None):
			self.layer = nn.Sequential(*layer)
		else:
			self.layer = layer
	def forward(self, x):
		if(self.layer is None):
			return x
		else:
			return self.layer(x)

class Net(nn.Module):
	def __init__(self, branch=None):
		super(Net, self).__init__()
		self.branch = Branch(branch)
	def forward(self, x):
		x = self.branch(x)
		return x




def calculate_entropy(y_pred):
  #m = nn.Softmax()	
  y_pred = torch.sigmoid(y_pred)
  #y_pred = m(y_pred)
  entropy = (-1)*torch.sum(torch.mul(y_pred, torch.log(y_pred)))
  return entropy


def calculate_entropy2(y_pred):
  y_pred = torch.sigmoid(y_pred)
  entropy = torch.sum(torch.mul(y_pred, torch.log(y_pred)))
  return entropy



def calculate_accuracy(fx, y):

  preds = fx.max(1, keepdim=True)[1].t()
  correct = preds.eq(y.view_as(preds)).sum()

  acc = correct.float()/preds.shape[0]

  return acc

def percentage_thresholds(lst, before_list):
  ctr = dict(collections.Counter(lst))

  for l in ctr.keys():
    percetage_exit = {l:float(ctr[l])/len(lst)}

  return percetage_exit

	

def exit_error(output_side_branch_list, target, error):
	v = 0
	for output_side_branch in output_side_branch_list:
		v += error(output_side_branch.cuda(), target)

	return v.to(device).float()

def to_one_hot(label):
	label = label.reshape(batch_size, 1)
	label = (label == torch.arange(n_classes).reshape(1, n_classes)).float()

	return label




