import os
import sys
import glob
import numpy as np
#from six.moves import cPickle as pickle
from scipy import linalg
import pickle


dirname = os.path.dirname(os.path.realpath(__file__))

def get_data(datasetPath):
    """
    Get train and test data from Cifar 10 dataset stored in datasetPath

    datasetPath: path of Cifar 10

    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
    with 6000 images per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000 images. 
    The test batch contains exactly 1000 randomly-selected images from each class. 
    The training batches contain the remaining images in random order, but some training batches 
    may contain more images from one class than another. 
    Between them, the training batches contain exactly 5000 images from each class.

    URL: https://www.cs.toronto.edu/~kriz/cifar.html

    Download : 
    1 - https://dl.dropboxusercontent.com/u/4542002/branchynet/data.npz
    2 - https://drive.google.com/file/d/0Byyuc5LmNmJPWUc5dVdUSms3U1E/view?usp=sharing

    """
    datasets = np.load(datasetPath)
    train_data = datasets['train_x']
    train_labels = datasets['train_y']
    test_data = datasets['test_x']
    test_labels = datasets['test_y']
    return train_data, train_labels, test_data, test_labels

def get_data_dev(numclasses=2):
    train_data, train_labels, test_data, test_labels = get_data()
    idx = (train_labels == 0)
    for i in range(1,numclasses):
        idx |= (train_labels == i)
    train_data = train_data[idx]
    train_labels = train_labels[idx]
    idx = (test_labels == 0)
    for i in range(1,numclasses):
        idx |= (test_labels == i)
    test_data = test_data[idx]
    test_labels = test_labels[idx]
    
    return train_data, train_labels, test_data, test_labels
