from __future__ import division
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import urllib
import scipy
import os

import pysmac
import pysmac.analyzer
import pysmac.utils

import sklearn.ensemble
import sklearn.datasets
import sklearn.cross_validation
from sklearn import metrics
from sknn.mlp import Classifier, Layer
from time import time

start = time()
n_iter = 300     ## Number of evaluations

## Dataset 9

url9 = "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
dataset9 = np.loadtxt(urllib.urlopen(url9), delimiter=",")
X = dataset9[:,0:4]
Y = dataset9[:,4]

# We split the data into training and test sets (70-30 respectively)
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X,Y, test_size=0.3, random_state=1)

# We fit the MLP with the hyperparameters given and return the model's accuracy on the testing set
def mlp(number_layers, number_neurons_1, number_neurons_2, number_neurons_3, number_neurons_4, 
	dropout_rate_1, dropout_rate_2, dropout_rate_3, dropout_rate_4, weight_decay, 
	activation_1, activation_2, activation_3, activation_4, learning_rate):

	layers = []
	number_neurons = []
	activation = []
	dropout = []

	number_neurons.append(number_neurons_1)
	number_neurons.append(number_neurons_2)
	number_neurons.append(number_neurons_3)
	number_neurons.append(number_neurons_4)

	activation.append(activation_1)
	activation.append(activation_2)
	activation.append(activation_3)
	activation.append(activation_4)

	dropout.append(dropout_rate_1)
	dropout.append(dropout_rate_2)
	dropout.append(dropout_rate_3)
	dropout.append(dropout_rate_4)

	for i in np.arange(number_layers):
		layers.append(Layer(activation[i], units=number_neurons[i], dropout = dropout[i], weight_decay = weight_decay))

	layers.append(Layer("Softmax",  units=2))
	
	predictor = Classifier(
    layers=layers,
    learning_rate=learning_rate,
    n_iter=25)

	predictor.fit(X_train, Y_train)
	
	return -metrics.accuracy_score(Y_test, predictor.predict(X_test))

# Here we define the hyperparameters search space from where we sample
parameter_definition=dict(\
		number_layers = ("integer", [1,4], 2),
		number_neurons_1  =("integer", [10,1000], 100, 'log'),
		number_neurons_2  =("integer", [10,1000], 100, 'log'),
		number_neurons_3  =("integer", [10,1000], 100, 'log'),
		number_neurons_4  =("integer", [10,1000], 100, 'log'),
		dropout_rate_1 =("real", [0,0.5],    0),
		dropout_rate_2 =("real", [0,0.5],    0),
		dropout_rate_3 =("real", [0,0.5],    0),
		dropout_rate_4 =("real", [0,0.5],    0),
		weight_decay =("real", [0,0.000031], 0, 'exp'),
		activation_1 = ("categorical", ['Rectifier', 'Tanh', 'Sigmoid'], 'Sigmoid'),
		activation_2 = ("categorical", ['Rectifier', 'Tanh', 'Sigmoid'], 'Sigmoid'),
		activation_3 = ("categorical", ['Rectifier', 'Tanh', 'Sigmoid'], 'Sigmoid'),
		activation_4 = ("categorical", ['Rectifier', 'Tanh', 'Sigmoid'], 'Sigmoid'),
		learning_rate = ("real", [0.001,0.01],    0.001)
		)

# We create the optimizer object
opt = pysmac.SMAC_optimizer( working_directory = './results/dataset9/c_smac/' % os.environ, persistent_files=True, debug = False)

# We set some parameters for the optimizer
value, parameters = opt.minimize(mlp,
					n_iter, parameter_definition,	# number of evaluations
					num_runs = 2,					# number of independent SMAC runs
					seed = 2,						# random seed
					num_procs = 2,					# two cores
					mem_limit_function_mb=1000,		# Memory limit
					t_limit_function_s = 10000	    # Time limit in seconds
					)
	
print(('The highest accuracy found: %f'%(-value)))
print(('Parameter setting %s'%parameters))
print("Bayesian Optimization took %.2f seconds for %d evaluations" % ((time() - start), n_iter))
