"""
Author: Roberto Pacheco
Date: 03/04/2020
Objective:  This script interacts to optimzation box. First step of the system.

"""


import torch, os, sys, config
import torch.nn as nn
import numpy as np
from ...branchMe.networks import alex_cifar10
import config
import pandas as pd
from ...branchMe.utils import read_parameters
from ...branchMe.partitioning_dnn.dnn_as_graphs import Model_DNN_Graph 


def generate_strategies_curves(i, return_dict):

	runtime_cloud_gpu_data = read_parameters(config.processing_time_cloud_path)
	runtime_cloud_cpu_data = read_parameters(config.processing_time_edge_path)
	output_size_data = 8*read_parameters(config.output_layers_size_path)/(10**6)

	processing_edge = read_parameters(config.processing_time_edge_path)


	accuracies_data = pd.read_json(config.accuracies_path)

	branchyNet = alex_cifar10.get_network()

	branchynet_graph = Model_DNN_Graph(branchyNet, runtime_cloud_gpu_data, 
		processing_edge, 
		output_size_data)


	df = pd.DataFrame(columns=("bandwidth", "t1", "t2", "t3", "t4", "probs", 
		"weighted_accuracy", "inference_time", "partitioning_layer"))

	cont = 0

	for bandwidth in config.bandwidth_list:
		print(bandwidth)
		for idx, row in accuracies_data.iterrows():
			weighted_acc_list, inference_time_list, partitioning_layer_list, probs = branchynet_graph.optim_acc_shortest_path(row, bandwidth)

			df.at[cont, "bandwidth"] = bandwidth
			df.at[cont, "t1"] = row.thresholds[0]
			df.at[cont, "t2"] = row.thresholds[1]
			df.at[cont, "t3"] = row.thresholds[2]
			df.at[cont, "t4"] = row.thresholds[3]

			df.at[cont, "probs"] = probs
			df.at[cont, "weighted_accuracy"] = weighted_acc_list
			df.at[cont, "inference_time"] = inference_time_list
			df.at[cont, "partitioning_layer"] = partitioning_layer_list
			cont+=1

		print("uauhuha")
		df.to_csv("a.csv")

	#return_dict["status"] = "ok"
