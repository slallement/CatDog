from __future__ import division, print_function, absolute_import
from skimage import color, io
import skimage.transform
from scipy.misc import imresize
import scipy.misc
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob
import time
import random

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import re
import model

import matplotlib.pyplot as plt

PATH_MODEL = 'model/'
PATH_IMG = '../img/input/'
NB_CATEGORIES = 3

def makeUniqueModel(filename):
	if not isfile(join(PATH_MODEL,filename)):
		return None
	m = re.match(r'(model_.*.tflearn).index',filename)
	if m:
		return m.group(1)
	return None

if __name__ == '__main__':
	size_image = 64
	test_image_path = '../img/test.jpg'
	
	# -----------------------------------
	# Load the model
	
	model_file = PATH_MODEL+sorted(list({makeUniqueModel(f) for f in listdir(PATH_MODEL)} - {None}))[-1]
	print('Model used: '+model_file)
	
	tf.reset_default_graph()
	
	# normalisation of images
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Create extra synthetic training data by flipping & rotating images
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)
	
	network = input_data(shape=[None, 64, 64, 3],
						 data_preprocessing=img_prep,
						 data_augmentation=img_aug)

	network = model.makeNetwork(network, NB_CATEGORIES)

	# Wrap the network in a model object
	model = tflearn.DNN(network, max_checkpoints = 10,
						tensorboard_verbose = 2, tensorboard_dir='tmp2/tflearn_logs/')
	
	
	model.load(model_file)
	print("done loading")
	
	#-------------------------------------------
	# predict cat/dog on the input image
	
	X_test = np.zeros((1, size_image, size_image, 3), dtype='float64')
	
	img = io.imread(test_image_path)
	lx, ly, lc = img.shape
	
	resize_factor = 1.0
	img = imresize(img, (int(lx/resize_factor), int(ly/resize_factor), 3))
	lx, ly, lc = img.shape
	
	result_data = np.zeros((lx, ly, 3), dtype=np.float32)

	img_size = 64
	pad = 8
	for iy in range(max(1,ly//pad-img_size//pad)):
		for ix in range(max(1,lx//pad-img_size//pad)):
			px = ix*pad
			py = iy*pad
			cropped_img = img[px:px+img_size,py:py+img_size,:lc]
			
			X_test[0] = np.array(cropped_img)
				
			result_y = model.predict(X_test)
			for y in range(py, py+img_size):
				for x in range(px, px+img_size):
					result_other = result_y[0][2]
					result_cat = result_y[0][0]
					result_dog = result_y[0][1]
					dist = abs(result_cat - result_dog)
					if result_other < 0.6 and dist > 0.6:
						if result_cat > result_dog:
							result_data[x,y,0] += 1.0
						else:
							result_data[x,y,1] += 1.0
	# normalize
	result_data[:,:,0] = result_data[:,:,0]/np.amax(result_data[:,:,0]) 
	result_data[:,:,1] = result_data[:,:,1]/np.amax(result_data[:,:,1]) 
	final_img = np.zeros((lx, ly, 3), dtype=np.int32)
	
	# save output
	for x in range(lx):
		for y in range(ly):
			for c in range(3):
				final_img[x,y,c] = result_data[x,y,0]*img[x,y,c]
	scipy.misc.imsave('./img/output_cat.jpg', final_img)
	
	for x in range(lx):
		for y in range(ly):
			for c in range(3):
				final_img[x,y,c] = result_data[x,y,1]*img[x,y,c]
	scipy.misc.imsave('./img/output_dog.jpg', final_img)
	