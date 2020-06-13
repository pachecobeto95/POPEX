import torch, os, sys, config
import torch.nn as nn
import numpy as np
from ...branchMe.networks import alex_cifar10
import config
import pandas as pd
from ...branchMe.utils import read_parameters
from ...branchMe.partitioning_dnn.dnn_as_graphs import Model_DNN_Graph 
import ast, json
#import psutil
import time
from threading import Thread
#import speedtest as st
from pythonping import ping

# Index : number of bytes sent
__BYTES_SENT__ = 0
# Index : number of bytes received
__BYTES_RECV__ = 1
# Index : number of packets sent
__PACKETS_SENT__ = 2
# Index : number of packets received
__PACKETS_RECV__ = 3
# Index : total number of errors while receiving
__ERR_IN__ = 4
# Index : total number of errors while sending
__ERR_OUT__ = 5
# Index : total number of incoming packets which were dropped
__DROP_IN__ = 6
# Index : total number of outgoing packets which were dropped
__DROP_OUT__ = 7




def __get_bytes_sent_total__():
	return psutil.net_io_counters(pernic=False)[__BYTES_SENT__]

def __get_bytes_recv_total__():
	return psutil.net_io_counters(pernic=False)[__BYTES_RECV__]	


def get_inc_points(accs, diffs, df_bandwidth, probs, inc_amt=-0.0005):
    
	accs = accs.apply(lambda x : np.mean(np.array(ast.literal_eval(x)))).values
	diffs = diffs.apply(lambda x : np.mean(ast.literal_eval(x))).values

	idxs = np.argsort(diffs)
	inc_accs = [accs[idxs[0]]]
	inc_diffs = [diffs[idxs[0]]]
	t1 = df_bandwidth.t1.values
	t2 = df_bandwidth.t2.values
	t3 = df_bandwidth.t3.values
	t4 = df_bandwidth.t4.values

	#inc_ts = [ts[idxs[0]]]
	inc_probs = [probs[idxs[0]]]
	for i, idx in enumerate(idxs[1:]):
		if accs[idx] > inc_accs[-1]+inc_amt:
			inc_accs.append(accs[idx])
			inc_diffs.append(diffs[idx])
			#inc_ts.append(ts[idx])
			inc_probs.append(probs[idx])

	return np.array(inc_accs), np.array(inc_diffs), np.array(inc_probs)

def choosing_strategy(current_bandwidth=1.1):

	df = pd.read_csv("a.csv")
	

	df_bandwidth = df[df.bandwidth == current_bandwidth]
	

	weighted_acc = df_bandwidth.weighted_accuracy
	inference_time = df_bandwidth.inference_time

	exit_probs = df.probs.values


	#inc_accs, inc_diffs, inc_exits = get_inc_points(weighted_acc, 
	#	inference_time, df_bandwidth, exit_probs)

	accs = np.array(weighted_acc.apply(lambda x : np.mean(np.array(ast.literal_eval(x)))))
	inf_time = np.array(inference_time.apply(lambda x : np.mean(np.array(ast.literal_eval(x)))))
	
	idx = np.zeros(inf_time.shape[0],dtype=bool)
	idx[inf_time < config.latency_requirement] = True

	choosen_accs = max(accs[idx])
	choosen_idx_strategy = np.argmax(accs[idx], axis=0)

	choosen_inf_time = inf_time[choosen_idx_strategy]
	choosen_partitioning_layer = np.array(df_bandwidth.partitioning_layer)[choosen_idx_strategy]

	return choosen_inf_time, choosen_accs, choosen_partitioning_layer


def send_decision(inf_time, acc, partitioning_layer):
	
	try:
		payload = {'inference_time': inf_time, "accuracy": acc,
		"partitioning_layer": partitioning_layer}

		url = "http://192.168.0.19:5000/api/a"

		r = requests.post(url, data=json.dumps(payload))
		print(r.status_code)

		return {"status": "ok"}
	except Exception as e:
		#print(e.args)
		return {"status": "error"}


def __monitorspeed__(destIP):
	
	response_list = ping(destIP, size=40, count=5)
	rtt_avg = response_list.rtt_avg_ms/2

	size_bits = 8*(config.default_size_packet)

	
	uplink_rate = size_bits/(rtt_avg*10**(-6))

	print(uplink_rate)

	return uplink_rate


def decision_maker():
	inf_time, acc, partitioning_layer = choosing_strategy(current_bandwidth=1.1)
	print("oi")
	#result = send_decision(inf_time, acc, partitioning_layer)
	print(__monitorspeed__("8.8.8.8"))

	return {"status":"ok"}





