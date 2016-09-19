
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
import urllib
import scipy
import os

import pysmac

import sklearn.ensemble
import sklearn.datasets
import sklearn.cross_validation
from sklearn import metrics
from sknn.mlp import Classifier, Layer


from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from time import time
from operator import itemgetter
import csv
from imblearn.combine import SMOTEENN

print(__doc__)

n_validations = 25     ## Number of Monte-Carlo Cross-Validations for each model's accuracy evaluated

## Dataset 11

url11 = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt"
dataset11 = np.genfromtxt(urllib.urlopen(url11))

X = dataset11[:,0:85]
Y = dataset11[:,85]

sm = SMOTEENN()
X, Y = sm.fit_sample(X, Y)

# Function to get indices from a random 70-30 split on all the data
def get_train_test_inds(y,train_proportion=0.7):

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds

# Function to report best scores from Random Search
def report(grid_scores, n_top=1):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

CViterator = []
for i in np.arange(n_validations):
    trainIndices,testIndices = get_train_test_inds(X,train_proportion=0.7)
    CViterator.append( (trainIndices, testIndices) )

start = time()

n_iter_search_1 =  40 # We search 40 random configurations with 1-hidden layer
n_iter_search_2 =  30 # We search 30 random configurations with 2-hidden layers
n_iter_search_3 =  30 # We search 30 random configurations with 3-hidden layers

## 1-layer

predictor1 = Classifier(
    layers=[
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Softmax",  units=2)],
    learning_rate=0.001,
    n_iter=25)

predictor1.fit(X, Y)

# Hyperparameters search space for a 1-hidden layer MLP
params={'dropout_rate': sp.stats.uniform(0, 0.5),
        'hidden0__units': sp.stats.randint(10, 1000) }

random_search1 = RandomizedSearchCV(predictor1,param_distributions=params,n_iter=n_iter_search_1, cv = CViterator,n_jobs=1)
random_search1.fit(X, Y)

## 2-layers

predictor2 = Classifier(
    layers=[
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Softmax",  units=2)],
    learning_rate=0.001,
    n_iter=25)

predictor2.fit(X, Y)

# Hyperparameters search space for a 2-hidden layers MLP
params={'dropout_rate': sp.stats.uniform(0, 0.5),
        'hidden0__units': sp.stats.randint(10, 1000),
        'hidden1__units': sp.stats.randint(10, 1000)}

random_search2 = RandomizedSearchCV(predictor2,param_distributions=params,n_iter=n_iter_search_2, cv = CViterator,n_jobs=1)
random_search2.fit(X, Y)

## 3-layers

predictor3 = Classifier(
    layers=[
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Softmax",  units=2)],
    learning_rate=0.001,
    n_iter=25)

predictor3.fit(X, Y)

# Hyperparameters search space for a 3-hidden layers MLP
params={'dropout_rate': sp.stats.uniform(0, 0.5),
        'hidden0__units': sp.stats.randint(10, 1000),
        'hidden1__units': sp.stats.randint(10, 1000),
        'hidden2__units': sp.stats.randint(10, 1000)}

random_search3 = RandomizedSearchCV(predictor3,param_distributions=params,n_iter=n_iter_search_3, cv = CViterator,n_jobs=1)
random_search3.fit(X, Y)


print("Best Model with 1-hidden layer:" )
report(random_search1.grid_scores_)

print("Best Model with 2-hidden layers:" )
report(random_search2.grid_scores_)

print("Best Model with 3-hidden layers:" )
report(random_search3.grid_scores_)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search_1 + n_iter_search_2 + n_iter_search_3))

# We save the accuracy for each configuration tried out on a .csv
with open('./results/dataset11/random_search/randomsearch.csv' % os.environ, 'w') as csvfile:
    fieldnames = ['parameters', 'score_mean', 'score_std']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for score in grid_search1.grid_scores_:
    with open('./results/dataset11/random_search/randomsearch.csv' % os.environ, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow({'parameters':  score.parameters, 
        'score_mean': score.mean_validation_score, 
        'score_std': np.std(score.cv_validation_scores)  })

for score in grid_search2.grid_scores_:
    with open('./results/dataset11/random_search/randomsearch.csv' % os.environ, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow({'parameters': score.parameters, 
        'score_mean': score.mean_validation_score, 
        'score_std': np.std(score.cv_validation_scores)  })

for score in grid_search3.grid_scores_:
    with open('./results/dataset11/random_search/randomsearch.csv' % os.environ, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow({'parameters': score.parameters, 
        'score_mean': score.mean_validation_score, 
        'score_std': np.std(score.cv_validation_scores)  })

