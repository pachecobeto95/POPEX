"""
Author: Roberto Pacheco
Date: 01/17/2019
Objetive: Model DNNs and BranchyNet as DAG.
"""

import networkx as nx, sys
import matplotlib.pyplot as plt
from .dnn_structures_graph import *
from typing import List, Set, Dict, Tuple, Optional,Callable
from networks.utils import Branch
import torch.nn as nn
import numpy as np
import itertools
import pandas as pd

class Model_DNN_Graph(object):
	"""A class used to intialize some DNNs as graphs"""

	def __init__(self, branchyNet, processing_time_cloud, processing_time_edge, output_size_data, cloud_edge_factor=None):
		"""Generates a generic branchyNet architecure, which is composed of a main branch and side branches
		inserted at intermediate layers
		
		Parameters
		-----------------------------------------
		n_layers: int 
			number of layers in main branch

		position_branches: List[int]
			position of side branches wil be inserted. 


		Output: A digraph object
			alexNet architecture modeled as a graph
		"""	

		self.processing_time_cloud = processing_time_cloud
				

		if(cloud_edge_factor is None):
			self.processing_time_edge = processing_time_edge

		else:
			self.cloud_edge_factor = cloud_edge_factor
			self.processing_time_edge = cloud_edge_factor*(self.processing_time_edge.mean())		


		self.output_size_data = output_size_data


		self.main_list  = branchyNet.main_list
		self.network = np.array(branchyNet.network)

		self.acc_main = 0.8
		self.dataset_size = 10000

		self.branchynet = self.__generate_graph()



	def __generate_graph(self):

		vertices = []
		branchynet = nx.DiGraph()

		j = 0


		#Filter all layers that are Conv2d or Branch.
		layers = list(filter(lambda layer : (isinstance(layer, nn.Conv2d) == True)
			or (isinstance(layer, Branch) == True) or (isinstance(layer, nn.Linear)), 
			self.network))


		#Create the nodes of graph with two different groups: main branch and side branch.
		for i, layer in enumerate(layers):
			if((isinstance(layer, nn.Conv2d) and i>0) or isinstance(layer, nn.Linear)):
				j+=1
				branchynet.add_node("l%s"%(j), group='main_branch')
			elif(isinstance(layer, Branch)):
				branchynet.add_node("b%s"%(j), group='side_branch')



		branchyNet_nodes = list(branchynet.nodes)

		#Mount a list of edges o side branches. 
		branchynet_edges = [(branchyNet_nodes[i], branchyNet_nodes[i+1]) 
		for i in range(len(branchyNet_nodes) - 1)]

		#Add the edges representing the side branches. 
		branchynet.add_edges_from(branchynet_edges)


		return branchynet


	def __pruning(self, acc_target):

		pruned_branchynet = self.branchynet.copy()

		
		self.position_branches = set(np.argwhere(~np.isnan(self.accuracies)).flatten()+1)

		branches_to_pruning = 1 + np.where(list(map(lambda branch_acc: branch_acc < acc_target, self.accuracies)))[0]

		self.position_branches -= set(branches_to_pruning)

		edge_to_pruning = list(filter(lambda x: "b%s"%(x[0]) in x[1], 
			itertools.product(branches_to_pruning, pruned_branchynet.edges)))
		

		if edge_to_pruning:
			edge_to_pruning = np.array(edge_to_pruning)[:, 1]

		pruned_branchynet.remove_edges_from(edge_to_pruning)
		
		pruned_branchynet.remove_nodes_from(["b%s"%b for b in branches_to_pruning])
		pruned_branchynet.add_edges_from([("l%s"%(b+1), "l%s"%(b+2)) for b in branches_to_pruning])

		return pruned_branchynet 

	def calculate_acc_avg(self):

		acc_branches = self.accuracies[np.array(list(self.position_branches)) - 1]
		num_exits = self.num_exits[np.array(list(self.position_branches)) - 1]
		weight_acc_branches = sum(acc_branches*(num_exits/self.dataset_size))
		weighted_acc_main = (1 - (sum(num_exits)/self.dataset_size))*self.acc_main
		

		#print(partitioning_layer)
		"""
		if(partitioning_layer=="output"):
			remain_branches = np.array(list(self.position_branches)) - 1
			acc_branches = self.accuracies[remain_branches]
			num_exits = self.num_exits[remain_branches]
			weight_acc_branches = sum(acc_branches*(num_exits/self.dataset_size))
			weighted_acc_main = (1 - (sum(num_exits)/self.dataset_size))*self.acc_main

		elif(partitioning_layer=="input"):
			weight_acc_branches = 0		
			weighted_acc_main = self.acc_main

		else:

			remain_branches = list(filter(lambda x: x<=int(partitioning_layer[-1]), list(self.position_branches)))
			if(len(remain_branches)==0):
				weight_acc_branches = 0
				weighted_acc_main = self.acc_main

			else:		
				remain_branches = np.array(remain_branches) - 1
				acc_branches = self.accuracies[remain_branches]
				num_exits = self.num_exits[remain_branches]
				weight_acc_branches = sum(acc_branches*(num_exits/self.dataset_size))
				weighted_acc_main = (1 - (sum(num_exits)/self.dataset_size))*self.acc_main

		"""
		return weight_acc_branches + weighted_acc_main


	def optim_acc_shortest_path(self, row_data, bandwidth):
		self.thresholds = np.array(row_data.thresholds)
		self.accuracies = np.array(row_data.accuracies)
		self.num_exits = np.array(row_data.num_exits)

		self.probs = self.num_exits/self.dataset_size
		
		accuracies = self.accuracies

		partitioning_layer_list = []
		weighted_acc_list = []
		inference_time_list = []
		#print(self.thresholds, accuracies)
		for acc in accuracies:

			pruned_branchynet = self.__pruning(acc)

			weighted_acc = self.calculate_acc_avg()

			current_shortest_graph = self.model_dnn_as_shortest_path_graph(pruned_branchynet, bandwidth)


			inference_time, shortest_path = nx.single_source_dijkstra(current_shortest_graph, 
				"input", "output", weight='weight')

			partitioning_layer = self.find_partitioning_layer(shortest_path)
			
			#weighted_acc = self.calculate_acc_avg(partitioning_layer)

			
			partitioning_layer_list.append(partitioning_layer)
			weighted_acc_list.append(weighted_acc)
			inference_time_list.append(inference_time)
			
		#print(self.processing_time_cloud)
		#print(self.processing_time_edge)
		#print(self.communication_time)


		#print(inference_time_list)
		#print(weighted_acc_list)
		#print(partitioning_layer_list)

		#sys.exit()
		
		return weighted_acc_list, inference_time_list, partitioning_layer_list, self.probs

	def __build_main_branch(self, n_layers:int):
		"""Generates a main branch of a generic branchynet"""
		main_branch = nx.DiGraph()
		layers = ["l%s"%(i) for i in range(1, n_layers+1)]
		edges = [("l%s"%(i), "l%s"%(i+1)) for i in range(1, n_layers)]
		main_branch.add_nodes_from(layers, group='main_branch')
		main_branch.add_edges_from(edges)

		return main_branch

	def __inserting_side_branches(self, main_branch_graph, position_branches: List[int]):
		n_layers = len(list(main_branch_graph.nodes()))

		if((max(position_branches) < n_layers)):
			side_branch_nodes = []
			side_branch_edges = []
			edge_remove_list = []

			for i in position_branches:
				side_branch_nodes.append("b%s"%(i))
				side_branch_edges.extend([("l%s"%(i), "b%s"%(i)), ("b%s"%(i), "l%s"%(i+1))])
				edge_remove_list.append(( "l%s"%(i), "l%s"%(i+1)))

			main_branch_graph.add_nodes_from(side_branch_nodes, group="side_branch")
			main_branch_graph.add_edges_from(side_branch_edges)
			main_branch_graph.remove_edges_from(edge_remove_list)
		

		return main_branch_graph

	def model_dnn_as_shortest_path_graph(self, branchyNet_graph, bandwidth):
		
		communication_time = self.output_size_data/bandwidth

		self.communication_time = communication_time

		dg = self.__generate_virtual_vertex(branchyNet_graph)


		prob_weights = self.generate_probabilities_graph(branchyNet_graph, 
			self.probs)

		#Set a dt 
		dt = 10**(-9)

		#Discover the input layer and output layer
		input_layer = list(dict(branchyNet_graph.in_degree()).keys())[list(dict(branchyNet_graph.in_degree()).values()).index(0)]
		output_layer = list(dict(branchyNet_graph.out_degree()).keys())[-1]
		

		#dg.add_edges_from([("input", "%s(e)"%(input_layer), {'weight':0}), 
		#	("input", "%s(c)"%(input_layer), {'weight': communication_time["input"]})])
		
		
		dg.add_edges_from([("input", "%s(e)"%(input_layer), {'weight':0}), 
			("input", "%s(c)"%(input_layer), {'weight': communication_time["input"]})])

				
		dg.add_edges_from([("%s(c)"%(output_layer), "output", {"weight":dt+
			(prob_weights[output_layer]*self.processing_time_cloud[output_layer])}),
		("%s(e)"%(output_layer), "output", 
			{"weigth":(prob_weights[output_layer])*self.processing_time_edge[output_layer]}) ])


		for node_id, node_att in dict(dg.nodes.data()).items():
			#print(node_id, node_att)

			if((node_att["group"] == "main_branch") and (node_att["device"] == "cloud")):

				fr = int(node_id[:-3][-1]) 
				to = "l%s(c)"%(fr + 1)

				if(to in dg.nodes()):
					dg.add_edge(node_id, to, weight= dt + (prob_weights["l%s"%(fr)]*self.processing_time_cloud["l%s"%(fr)]))

			if((node_att["group"] == "main_branch") and (node_att["device"] == "edge") and (int(node_id[:-3][-1]) < len(branchyNet_graph.nodes)-1) ):
				fr = int(node_id[:-3][-1])

				if("l%s"%(fr) != output_layer):
					dg.add_edge(node_id, "l%s(aux)"%(fr), weight=prob_weights["l%s"%(fr)]*self.processing_time_edge["l%s"%(fr)])


			if((node_att["group"] == "aux") and (node_att["device"] == "edge") and (int(node_id[1]) < len(branchyNet_graph.nodes)-1)):

				fr_id = int(node_id[1])

				if("l%s(c)"%(fr_id+1) in dg.nodes()):
					#edge cost is communication delay
					dg.add_edge(node_id, "l%s(c)"%(fr_id+1), weight=dt+prob_weights["l%s"%(fr)]*communication_time["l%s"%(fr)])


				if((fr_id-1 in self.position_branches) and ("b%s(e)"%(fr_id) in dg.nodes)):
					#edge cost is zero delay
					dg.add_edge(node_id, "b%s(e)"%(fr_id), weight=0)

				else:
					#edge cost is zero delay
					if("l%s(e)"%(fr_id+1) in dg.nodes()):
						dg.add_edge(node_id, "l%s(e)"%(fr_id+1), weight=0)

			if((node_att["group"] == "side_branch") and (node_att["device"] == "edge")):
				#edge cost is processing time of side branch
		
				dg.add_edge(node_id, "l%s(e)"%(fr_id+1), weight=prob_weights["b%s"%(fr_id)]*self.processing_time_edge["b%s"%(fr_id)])
		
		
		return dg

	def find_partitioning_layer(self, shortest_path):
		partition_layer = "output"
		
		for i, node in enumerate(shortest_path):
			index_partition = node.find("(c)")

			if(index_partition != -1 ):					

				
				if(int(node[index_partition-1])-1 == 0):
					partition_layer = "input"
				
				else:
					partition_layer = shortest_path[i-1][:2]
				
				break



		return partition_layer


	def __generate_virtual_vertex(self, branchyNet_graph):
		

		#dg = branchyNet_graph.__class__()
		#dg.add_nodes_from(branchyNet_graph)

		dg = nx.DiGraph()
		dg.add_nodes_from(["input", "output"], group="s-t")

		for node_id, node_att  in branchyNet_graph.nodes.data():
			
			if(node_att["group"]=="main_branch"):

				dg.add_node("%s(c)"%(node_id), group=node_att["group"], device="cloud")
				dg.add_node("%s(e)"%(node_id), group=node_att["group"], device="edge")
				
				if(int(node_id.split("l")[-1]) < len(branchyNet_graph.nodes)-1):
					dg.add_node("%s(aux)"%(node_id), group="aux", device="edge")

			if(node_att["group"]=="side_branch"):
				dg.add_node("%s(e)"%(node_id), group=node_att["group"], device="edge")
		
		return dg


	def alexNet(self):
		"""Generates a digraph based on alexnet architecture"""
		self.alexNet_graph = nx.DiGraph()
		H = nx.path_graph(self.n_vertex)
		self.alexNet_graph.add_nodes_from(H, group="main_branch")
		self.alexNet_graph.add_edges_from(H.edges())

		return self.alexNet_graph

	def squeezeNet(self):
		"""Generates a digraph based on alexnet architecture"""
		return nx.DiGraph(self.squeezeNet_layers_dict)

	def b_alexNet(self):
		"""Generates a digraph based on b_squeezeNet architecture"""

		alexNet_graph = self.alexNet()
		if(self.position_branches is not None):
			for k in self.position_branches:
				alexNet_graph.add_node("b%s"%(k), group="side_branch")
				alexNet_graph.add_edge(k, "b%s"%(k))
				#alexNet_graph.add_edge("b%s"%(k), k+1)


		return alexNet_graph



	def show_graph(self, graph):
		"""Creates a graphic representation of a graph

		graph: object graph
		"""
		nx.draw(graph, with_labels=True)
		plt.draw()
		plt.show()				



	def generate_probabilities_graph(self, branchynet, prob):
				
		prob_dict = {layer:1.0 for layer in branchynet.nodes}
		

		position_branches = list(self.position_branches)

		if position_branches:

			for i, (branches, p) in enumerate(zip(position_branches, prob), 1):
				#branches += 1
				for layer in range(1, (len(branchynet.nodes)-sum(~np.isnan(self.accuracies))) + 1):
					
					if(layer < branches ):

						prob_dict["l%s"%(layer)] *= 1

					elif(layer == branches ):	
						
						prob_dict["b%s"%branches] *= p

					else:
						prob_dict["l%s"%layer] *= (1 - p)
						if("b%s"%(layer) in list(prob_dict.keys())):
							prob_dict["b%s"%(layer)] *= 1 - p

		return prob_dict















