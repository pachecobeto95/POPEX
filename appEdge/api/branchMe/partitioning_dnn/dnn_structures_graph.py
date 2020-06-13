"""DNN architecures described as digraphs."""


alexnet_structure = {"conv1": "conv2", "conv2":"conv3",
			"conv3":"conv4", "conv4": "conv5", "conv5":"fc1", "fc1":"fc2", "fc2":"fc3"}


squeezeNet_structure_with_one_branch = {"conv1": ["max1"], "max1":["fire1"],
	"fire1": ["fire2"],"fire2":["fire3"], 
	"fire3": ["fire4"], "fire4":["fire5"], "fire5":["fire6"], "fire6":["fire7"], "fire7":["fire8"],
	"fire8":"conv_final"}


b_squeezeNet_structure_with_one_branch = {"conv1": ["max1"], "max1":["fire1"], 
	"fire1":["branch1"], "fire1": ["fire2"],"fire2":["fire3"], 
	"fire3": ["fire4"], "fire4":["fire5"], "fire5":["fire6"], "fire6":["fire7"], "fire7":["fire8"],
	"fire8":"conv_final"}


 


