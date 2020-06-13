import requests, sys, json, os


def sendImg(url, imgFile, fileName):
	try:

		files = {'media': open(imgFile, 'rb')}

		r = requests.post(url, files=files)

		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
		sys.exit()
	else:
		print("upload achieved")




def sendJson(url, jsonData):
	try:
		r = requests.post(url, json=jsonData)
		if(r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s'%(r.status_code))

	except Exception as err:
		print(err.args)
		
	else:
		print("upload achieved")
