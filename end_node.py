import os, config
from routines.sendData import sendImg
import requests, sys, json, os
#import cv2



"""
End device routine:
Send an image
"""
def sendimage():
	try:

		file_test = "dog.jpg"
		imgFile = os.path.join(config.DIR_NAME, 
			file_test)

		url = config.URL_EDGE + "/api/edgearch/edge"

		sendImg(url, imgFile, file_test)
	except Exception as e:
		print(e.args)


def sendNetwork():
	try:
		network = "b_alexNet"
		payload = {'network': network}

		url = config.URL_EDGE + "/api/edgearch/initiate_edge"
		print(payload)

		r = requests.post(url, data=json.dumps(payload))


	except Exception as e:
		print(e.args)





if __name__ == "__main__":
	#sendNetwork()
	sendimage()

    