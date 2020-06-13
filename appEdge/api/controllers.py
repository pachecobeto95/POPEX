from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import edgeProcessing
from .services.background import extract_parameters
from .services.decision_maker import generate_multiples_strategies, making_decision
import logging, json, os, time, sys, config
#import cv2
from config import save_images_path
from multiprocessing import Process, Manager, Queue



api = Blueprint("api", __name__, url_prefix="/api")

def worker(i, return_dict):
	'''worker function'''
	print (str("status") + ' represent!')
	return_dict["status"] = "ok"

#@api.before_app_first_request
def test():
	manager = Manager()
	return_dict = manager.dict()
	jobs = []

	if not os.path.exists(config.benchmark_processing_layers_edge):
		p0 = Process(extract_parameters.extract_processing_time)
		p0.start()

	p = Process(target=generate_multiples_strategies.generate_strategies_curves, args=(1, return_dict))
	jobs.append(p)
	p.start()

	return jsonify({"status": "ok"}), 200

	if(return_dict["status"] == 'ok'):
		return jsonify({"status": "ok"}), 200
	else:
		return jsonify({"status": "ok"}), 500



@api.route('/edgearch/edge', methods = ['POST'])
def edge():
	fileImg = request.files["media"]

	result = making_decision.decision_maker(fileImg)

	if(result["status"] == 'ok'):
		return jsonify({"status": "ok"}), 200
	else:
		return jsonify({"status": "ok"}), 500

	

@api.route('/edgearch/initiate_edge', methods = ['POST'])
def initiate_edge():
	network = request.get_json(force=True)

	manager = Manager()
	return_dict = manager.dict()
	jobs = []

	if not os.path.exists(config.benchmark_processing_layers_edge):
		p0 = Process(extract_parameters.extract_processing_time)
		p0.start()

	p = Process(target=generate_multiples_strategies.generate_strategies_curves, args=(1, return_dict))




	return jsonify({"ok":"generate multiple strategies"}), 200


	if not os.path.exists(config.benchmark_processing_layers_edge):
		result = extract_parameters.extract_processing_time()


	return jsonify({"ok":"generate multiple strategies"}), 200


	#generate_multiples_strategies()
	
	return jsonify({"ok":"generate multiple strategies"}), 200


@api.route('/edgearch/setedgecache', methods = ['POST'])
def setedgecache():

	fileImg = request.json
	print(fileImg)
	sys.exit()
	result = edgeProcessing.setCache(fileImg)

	if(result['status'] == 'ok'):
		return jsonify(result), 200
	else:
		return jsonify(result), 500
