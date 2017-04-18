
from __future__ import division, print_function, absolute_import
import os
from glob import glob
import time
from scipy.misc import imresize
import numpy as np
from skimage import color, io
import skimage.transform
from sklearn.cross_validation import train_test_split

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
import operator
from functools import reduce

import model

# the input images folder : it should contain 3 folders : cat, dog and other
FILE_PATH = "../img/input/"

# limit the training time
EPOCH_MAX = 100

if __name__ == '__main__':
	print("Begining")
	files_path = FILE_PATH
	
	limit = 5000 # limit the number of images to read in each category
	size_image = 64 # the images will be resized to this size
	
	# get all the images names
	other_files = [files_path+'other/'+f for f in sorted(os.listdir( files_path+'other/'))[:limit]]
	cat_files = [files_path+'cat/'+f for f in sorted(os.listdir( files_path+'cat/'))[:limit]]
	dog_files = [files_path+'dog/'+f for f in sorted(os.listdir( files_path+'dog/'))[:limit]]
	
	list_data = [cat_files,dog_files, other_files]
	nb_categories = len(list_data)
	all_images = reduce(operator.add, list_data)
	all_labels = [ [i]*len(images) for i, images in enumerate(list_data)]
	all_labels = reduce(operator.add, all_labels)
	n_files = len(all_images)
	
	# load images
	allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
	allY = np.zeros(n_files)
	count = 0
	print('Loading images, it can take a while')
	for f, label_id in zip(all_images, all_labels):
		try:
			img = io.imread(f)
			if len(img.shape) < 3:
				img = color.gray2rgb(img)
			new_img = imresize(img, (size_image, size_image, 3))
			allX[count] = np.array(new_img)
			allY[count] = label_id
			count += 1
		except Exception as e:
			print('Error on file '+f)
			print("{0}".format(e))
			continue
	
	print('Images loaded')
	
	# Prepreocessing

	# test-train split   
	X, X_test, Y, Y_test = train_test_split(allX, allY, test_size=0.1, random_state=42)
	Y = to_categorical(Y, nb_categories)
	Y_test = to_categorical(Y_test, nb_categories)

	# normalisation of images
	img_preprocessing = ImagePreprocessing()
	img_preprocessing.add_featurewise_zero_center()
	img_preprocessing.add_featurewise_stdnorm()

	# Create extra training data by translating and rotating images
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)

	# Neural network
	network = input_data(shape=[None, size_image, size_image, 3],
						 data_preprocessing=img_preprocessing, data_augmentation=img_aug)

	network = model.makeNetwork(network, nb_categories)

	model = tflearn.DNN(network, checkpoint_path='model/checkpoint-'+time.strftime("%Y-%m-%d-%H-%M-%S")+'.tflearn', max_checkpoints = 10,
						tensorboard_verbose = 2, tensorboard_dir='board/tflearn_logs/')

	# train the model
	model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=256,
		  n_epoch=EPOCH_MAX, run_id='snapshot_'+time.strftime("%Y-%m-%d-%H-%M-%S"), show_metric=True, snapshot_epoch=True)

	# save the model
	model.save('model/model_'+time.strftime("%d-%H-%M-%S")+'.tflearn')