"""
Author: Roberto Pacheco
Date: 03/04/2020
Objective:  This script interacts to optimzation box. First step of the system.

"""


import torch, os, sys
import torch.nn as nn
import numpy as np
from networks import alex_cifar10
#from datasets import pcifar10
#from branchynet import utils, visualize
#import the hyperparameters
import config
import pandas as pd
from utils import read_parameters
from partitioning_dnn.dnn_as_graphs import Model_DNN_Graph 



dirname = os.path.dirname(os.path.realpath(__file__))


runtime_layers_gpu_path = os.path.join(dirname, "data_results", "current_layers_time_using_COLAB_gpu.csv")
runtime_layers_cpu_path = os.path.join(dirname, "data_results", "current_layers_time_using_COLAB_cpu.csv")
output_layers_size_path = os.path.join(dirname, "data_results", "b_alexNet_bytes_layers.csv")
save_path = os.path.join(dirname, "data_results", "COLAB_WEIGHTED_AFTER_optim_box_surgery_with_CPU.json")
accuracies_path = os.path.join(dirname, "data_results", "FINAL_FINAL_save_eval_thesholds_cifar_10.json")




runtime_cloud_gpu_data = read_parameters(runtime_layers_gpu_path)
runtime_cloud_cpu_data = read_parameters(runtime_layers_cpu_path)
output_size_data = 8*read_parameters(output_layers_size_path)/(10**6)

"""
accuracies_data columns:
['thresholds', 'overall_acc', 'overall_final', 'totaltime', 'num_exits',
       'numexits_keep', 'acc_dual_models', 'accbreakdowns', 'accuracies',
       'diff_branches', 'exit_ratio']
"""

branchyNet = alex_cifar10.get_network()
	

accuracies_data = pd.read_json(accuracies_path).thresholds


print(accuracies_data)

sys.exit()



bandwidth_list  = [0.1, 1.1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.75, 4, 5.5, 7, 10, 13, 18.8, 20]
edge_cloud_factor_list = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 50, 100, 500, 1000, 1500, 2000, 3000, 5000]


#df = pd.read_json(save_path)
#cont = len(df) + 1




df = pd.DataFrame(columns=("bandwidth","edge_cloud_factor", "thresholds", "probs", 
	"weighted_accuracy", "inference_time", "partitioning_layer"))

cont = 0

for factor in edge_cloud_factor_list:

	print("Factor Edge-Cloud: %s"%(factor))
	
	branchynet_graph = Model_DNN_Graph(branchyNet, runtime_cloud_gpu_data, factor, 
		output_size_data)

	for bandwidth in bandwidth_list:
		print("Bandwidth: %s"%(bandwidth))
		
		for idx, row in accuracies_data.iterrows():

			weighted_acc_list, inference_time_list, partitioning_layer_list, probs = branchynet_graph.optim_acc_shortest_path(row, bandwidth)


			df.at[cont, "bandwidth"] = bandwidth
			df.at[cont, "edge_cloud_factor"] = factor
			df.at[cont, "thresholds"] = row.thresholds
			df.at[cont, "probs"] = probs
			df.at[cont, "weighted_accuracy"] = weighted_acc_list
			df.at[cont, "inference_time"] = inference_time_list
			df.at[cont, "partitioning_layer"] = partitioning_layer_list
			cont+=1


		df.to_json(save_path)

