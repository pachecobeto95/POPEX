from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import cloudProcessing
import logging, json, os, time, sys, config
import logging, json, os, time, sys, config
#import cv2
from config import save_images_path
from multiprocessing import Process, Manager, Queue



api = Blueprint("api", __name__, url_prefix="/api")

@api.route('/a', methods = ['POST'])
def edgearch_cloud():
	print("uahuaha")
	fileImg = request.get_json(force=True)
	print(fileImg)

	
	result = {"status": "ok"}
	
	if(result['status'] == 'ok'):
		return jsonify(result), 200
	else:
		return jsonify(result), 500
