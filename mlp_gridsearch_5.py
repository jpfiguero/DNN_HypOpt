
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.grid_search import GridSearchCV
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

import logging 
logging.basicConfig()

import csv

print(__doc__)

n_validations = 25     ## Number of Monte-Carlo Cross-Validations for each model's accuracy evaluated

# Dataset 5

url5 = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
dataset5 = np.loadtxt(urllib.urlopen(url5), delimiter=",")
X = dataset5[:,0:57]
Y = dataset5[:,57]

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

start = time()

CViterator = []
for i in np.arange(n_validations):
    trainIndices,testIndices = get_train_test_inds(X,train_proportion=0.7)
    CViterator.append( (trainIndices, testIndices) )

## 1-hidden layer

predictor1 = Classifier(
    layers=[
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Softmax",  units=2)],
    learning_rate=0.001,
    n_iter=25)

predictor1.fit(X, Y)

# The grid we try for a 1-hidden layer MLP (6x6 = 36 combinations)
param_grid1 = {'dropout_rate': [0, 0.05, 0.10, 0.15, 0.20, 0.25],
        'hidden0__units': [32, 64, 128, 256, 512, 1024]}

grid_search1 = GridSearchCV(predictor1, param_grid=param_grid1, cv = CViterator,n_jobs=8)
grid_search1.fit(X, Y)

## 2-hidden layers

predictor2 = Classifier(
    layers=[
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Softmax",  units=2)],
    learning_rate=0.001,
    n_iter=25)

predictor2.fit(X, Y)

# The grid we try for a 2-hidden layers MLP (4x3x3 = 36 combinations)
param_grid2 = {'dropout_rate': [0, 0.05, 0.10, 0.25],
        'hidden0__units': [128, 512, 1024],
        'hidden1__units': [128, 512, 1024]}

grid_search2 = GridSearchCV(predictor2, param_grid=param_grid2, cv = CViterator,n_jobs=8)
grid_search2.fit(X, Y)

## 3-hidden layers

predictor3 = Classifier(
    layers=[
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Sigmoid", units=100, dropout = 0),
    Layer("Softmax",  units=2)],
    learning_rate=0.001,
    n_iter=25)

predictor3.fit(X, Y)

# The grid we try for a 3-hidden layers MLP (3x2x3x2 = 36 combinations)
param_grid3 = {'dropout_rate': [0, 0.10, 0.25],
        'hidden0__units': [128, 512],
        'hidden1__units': [128, 512, 1024],
        'hidden2__units': [128, 512]}

grid_search3 = GridSearchCV(predictor3, param_grid=param_grid3, cv = CViterator,n_jobs=8)
grid_search3.fit(X, Y)

print("Best Model with 1-hidden layer:" )
report(grid_search1.grid_scores_)

print("Best Model with 2-hidden layers:" )
report(grid_search2.grid_scores_)

print("Best Model with 3-hidden layers:" )
report(grid_search3.grid_scores_)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search1.grid_scores_) + len(grid_search2.grid_scores_) + len(grid_search3.grid_scores_)))

# We save the accuracy for each configuration tried out on a .csv
with open('./results/dataset5/grid_search/gridsearch.csv' % os.environ, 'w') as csvfile:
    fieldnames = ['parameters', 'score_mean', 'score_std']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for score in grid_search1.grid_scores_:
    with open('./results/dataset5/grid_search/gridsearch.csv' % os.environ, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow({'parameters':  score.parameters, 
        'score_mean': score.mean_validation_score, 
        'score_std': np.std(score.cv_validation_scores)  })

for score in grid_search2.grid_scores_:
    with open('./results/dataset5/grid_search/gridsearch.csv' % os.environ, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow({'parameters': score.parameters, 
        'score_mean': score.mean_validation_score, 
        'score_std': np.std(score.cv_validation_scores)  })

for score in grid_search3.grid_scores_:
    with open('./results/dataset5/grid_search/gridsearch.csv' % os.environ, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow({'parameters': score.parameters, 
        'score_mean': score.mean_validation_score, 
        'score_std': np.std(score.cv_validation_scores)  })

