import tflearn

from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.metrics import Accuracy
from tflearn.layers.estimator import regression

def makeNetwork(network, nb_categories):
	# architecture of the network
	network = conv_2d(network, 32, 3, activation='relu', name='conv_1')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 64, 3, activation='relu', name='conv_2')
	network = conv_2d(network, 64, 3, activation='relu', name='conv_3')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 128, 3, activation='relu', name='conv_4')
	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.4)
	
	network = fully_connected(network, nb_categories, activation='softmax')
	
	# how the network will be trained
	acc = Accuracy(name="Accuracy")
	network = regression(network, optimizer='adam',
						 loss='categorical_crossentropy',
						 learning_rate=0.0009, metric=acc)
	return network