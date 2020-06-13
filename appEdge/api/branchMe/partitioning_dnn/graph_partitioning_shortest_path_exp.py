"""
Author: Roberto Pacheco
Date: 01/21/2019
Objetive: Experiments with BranchyNet as DAG.
"""

import numpy as np, sys
import pandas as pd
from dnn_as_graphs import Model_DNN_Graph
from itertools import product
import networkx as nx
import argparse


def write_results(cont,df_processing_time_edge,df_communication_time,df_inference_time, partitioning_layer):
	for proc_edge, com_time, layer in zip(processing_time_edge, communication_time,
		layer_list):
				
		df_processing_time_edge.loc[cont, layer] = proc_edge
		df_communication_time.loc[cont, layer] = com_time

		df_processing_time_edge.loc[cont, "inference_time"] = inference_time
		df_communication_time.loc[cont, "inference_time"] = inference_time

		df_processing_time_edge.loc[cont, "edge_cloud_factor"] = edge_cloud_factor
		df_processing_time_edge.loc[cont, "p0"] = probs_tuple
		df_processing_time_edge.loc[cont, "inference_time"] = inference_time

		df_communication_time.loc[cont, "bandwidth"] = bandwidth
		df_communication_time.loc[cont, "p0"] = probs_tuple
		df_communication_time.loc[cont, "inference_time"] = inference_time


		df_inference_time.loc[cont, "bandwidth"] = bandwidth
		df_inference_time.loc[cont, "edge_cloud_factor"] = edge_cloud_factor
		df_inference_time.loc[cont, "p0"] = probs_tuple
		df_inference_time.loc[cont, "inference_time"] = inference_time
		df_inference_time.loc[cont, "partitioning_layer"] = partitioning_layer

		df_inference_time.to_csv("%s_inference.csv"%(save_path))
		df_communication_time.to_csv("%s_communication.csv"%(save_path))
		df_processing_time_edge.to_csv("%s_processing.csv"%(save_path))


def read_parameters(pd_path):
	df = pd.read_csv(pd_path)
	df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
	mean_df = df.mean()
	return mean_df


def generate_prob(prob_list, num_exits):
	ps = list(product(*([prob_list]*(num_exits))))
	return ps

def generate_probabilities_graph(n_layers, position_branches, prob):
	
	prob_final = np.ones(n_layers)
	prob_dict = {}

	position_branches = np.array(position_branches)
	prob = np.array(prob)

	for i, (branches,p) in enumerate(zip(position_branches, prob), 1):
		for layer in range(n_layers):
			#print(layer, branches)
			if(layer < branches ):

				prob_final[layer] *= 1

			elif(layer == branches ):	

				prob_dict["b%s"%layer] = 1 - p

			else:
				prob_final[layer] *= (1 - p)
				
					
			prob_dict["l%s"%layer] = prob_final[layer]

	return prob_dict


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-n', '--network')
parser.add_argument('-b', '--branches', nargs='*', type=int, default=[0,1])

args = parser.parse_args()
network_layers = {"b_alexNet":8, 
"b_squeezeNet": 10}
if(args.network not in network_layers.keys()):
	print("Error: no network ")
	sys.exit()
	
	


processing_time_layers_path = "%s_time_layers_with_gpu.csv"%(args.network)
output_layers_size_path = "%s_bytes_layers.csv"%(args.network)
save_path = "%s_surgery_exp_with_gpu"%(args.network)

n_layers = network_layers[args.network]

position_branches = args.branches

bandwidth_initial = 1
bandwidth_final  = 100
bandwidth_step  = 5
prob_step = 0.1

bandwidth_list = [1.1, 5.85, 18.8]

#edge_cloud_factor_list = [5, 10, 20, 30, 50] + [100, 200, 300, 400, 500, 1000]



prob_list = np.arange(0, 1+prob_step, prob_step)


if(len(position_branches) > 1):
	prob_list = generate_prob(prob_list, len(position_branches))


processing_time_cloud = read_parameters(processing_time_layers_path)

output_layers_size = 8*(read_parameters(output_layers_size_path))/(10**6)


dg = Model_DNN_Graph(n_layers, position_branches)
b_alexNet_graph = dg.b_alexNet()


df_communication_time = pd.DataFrame(columns = ("bandwidth","edge_cloud_factor","p0",
	"conv1","conv2","conv3",
	"conv4","conv5","exit1","exit2", "fc1","fc2","fc3", "inference_time"))

df_processing_time_edge = pd.DataFrame(columns = ("edge_cloud_factor","inference_time"))

df_inference_time= pd.DataFrame(columns = ("bandwidth","edge_cloud_factor","p0",
	"inference_time", "partitioning_layer"))

layer_list = ["conv1","conv2","conv3","conv4","conv5","exit1","exit2", "fc1","fc2","fc3"]

#cont = 1

df_inference = pd.read_csv("b_alexNet_surgery_exp_with_gpu_inference.csv")


#edge_cloud_factor_list = [10, 20, 30, 40, 50, 100]
edge_cloud_factor_list = [10, 50, 100, 200, 300, 400, 500, 1000]

cont = len(df_inference) + 1

for probs_tuple in prob_list:
	print("Evaluating with prob: %s" %(probs_tuple))

	prob_dict = generate_probabilities_graph(n_layers, position_branches, [probs_tuple])

	for bandwidth in bandwidth_list:
		print("Bandwidth: %s"%(bandwidth))
		communication_time = output_layers_size/bandwidth


		for edge_cloud_factor in edge_cloud_factor_list:
			print("Difference between edge and cloud: %s"%(edge_cloud_factor))
			processing_time_edge = (edge_cloud_factor)*processing_time_cloud
			edge_time = (edge_cloud_factor)*processing_time_cloud.sum()			

			b_alexNet_sp_graph = dg.model_dnn_as_shortest_path_graph(b_alexNet_graph, processing_time_cloud, 
				processing_time_edge, communication_time, prob_dict)
			
			
			inference_time, shortest_path = nx.single_source_dijkstra(b_alexNet_sp_graph, "input", "output", weight='weight')
			
			
			partitioning_layer = dg.find_partitioning_layer(shortest_path)

			df_inference.loc[cont, "bandwidth"] = bandwidth
			df_inference.loc[cont, "edge_cloud_factor"] = edge_cloud_factor
			df_inference.loc[cont, "p0"] = probs_tuple
			df_inference.loc[cont, "inference_time"] = inference_time
			df_inference.loc[cont, "partitioning_layer"] = partitioning_layer

			df_processing_time_edge.loc[cont, "edge_cloud_factor"] = edge_cloud_factor
			df_processing_time_edge.loc[cont, "inference_time"] = edge_time


			df_inference.to_csv("b_alexNet_surgery_exp_with_gpu_inference.csv")
			#df_processing_time_edge.to_csv("b_alexNet_surgery_edge_processing.csv")

			#write_results(cont,df_processing_time_edge,df_communication_time,
			#	df_inference_time, partitioning_layer)

			
			cont += 1
			






