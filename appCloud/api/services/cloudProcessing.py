from flask import jsonify, session, current_app as app
#from .featureExtractionCloud import FeatureExtractor
#from cache import LFUCache
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
#import lfucache.lfu_cache as lfu_cache
from sklearn import linear_model
from scipy.misc import derivative




def receiveData(fileImg):
	try:
		url = config.URL_SET_EDGE + "/api/edgearch/edgesetcheck"
		uploadJsonData(url, fileImg.filename)

		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}

def uploadImgData(fileImg):
	try:
		imgPath = __saveImage(fileImg)

		url = config.URL_SET_EDGE + "/api/edgearch/edgesetcache"
		uploadImg(url, fileImg)

		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}


def uploadJsonData(url, imgName):
	try:
		jsonData = {"name": imgName}
		r = requests.post(url, json=jsonData)

		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
		sys.exit()
	else:
		print("upload achieved")


def uploadImg(url, fileImg):
	try:
		imgPath = os.path.join(config.DIR_NAME, "appEdge", "api", "edgeDataset", fileImg.filename)
		files = {'file': (fileImg.filename, open(imgPath, 'rb'), 'image/x-png')}
		r = requests.post(url, files=files)

		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
		sys.exit()
	else:
		print("upload achieved")


def __saveImage(imgFile):
	imgPath = os.path.join(config.DIR_NAME, "appCloud", "api", "cloudDataset", imgFile.filename)
	if(imgFile):
		imgFile.save(imgPath)
	return imgPath

