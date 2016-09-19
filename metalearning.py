from __future__ import division
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import urllib
import scipy
import os

from collections import defaultdict, OrderedDict, deque
import copy
import sys

import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
import sklearn
# TODO use balanced accuracy!
import sklearn.metrics
import sklearn.cross_validation
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier
from imblearn.combine import SMOTEENN

import csv

# We implement the Meta-features for all the datasets
# This is based in the work done by Feurer et al (2015)
# We implement the same meta-features extracted by them
# except the land-marking ones
# Most part of this code is based in the code found in: 
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/metalearning/metafeatures/metafeatures.py

# We define the function for extracting the Kurtosis of all the columns of a matrix
def Kurtosisses(X):
    kurts = []
    for i in range(X.shape[1]):
    	kurts.append(scipy.stats.kurtosis(X[:, i].astype(np.float)))

    return kurts

# We define a function for extracting the minimum Kurtosis of all the columns of a matrix
def KurtosisMin(X):
    kurts = Kurtosisses(X)
    minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
    return minimum if np.isfinite(minimum) else 0

# We define a function for extracting the maximum Kurtosis of all the columns of a matrix
def KurtosisMax(X):
    kurts = Kurtosisses(X)
    maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
    return maximum if np.isfinite(maximum) else 0

# We define a function for extracting the mean Kurtosis of all the columns of a matrix
def KurtosisMean(X):
    kurts = Kurtosisses(X)
    mean = np.nanmean(kurts) if len(kurts) > 0 else 0
    return mean if np.isfinite(mean) else 0

# We define a function for extracting the standard deviation of all the Kurtosisses extracted from all the columns of a matrix
def KurtosisSTD(X):
    kurts = Kurtosisses(X)
    std = np.nanstd(kurts) if len(kurts) > 0 else 0
    return std if np.isfinite(std) else 0

# We define a function for extracting the minimum Skewness of all the columns of a matrix
def Skewnesses(X):
    skews = []
    for i in range(X.shape[1]):
        skews.append(scipy.stats.skew(X[:, i].astype(np.float)))

    return skews

# We define a function for extracting the minimum Skewness of all the columns of a matrix
def SkewnessMin(X):
    skews = Skewnesses(X)
    minimum = np.nanmin(skews) if len(skews) > 0 else 0
    return minimum if np.isfinite(minimum) else 0

# We define a function for extracting the maximum Skewness of all the columns of a matrix
def SkewnessMax(X):
    skews = Skewnesses(X)
    maximum = np.nanmin(skews) if len(skews) > 0 else 0
    return maximum if np.isfinite(maximum) else 0

# We define a function for extracting the mean Skewness of all the columns of a matrix
def SkewnessMean(X):
    skews = Skewnesses(X)
    mean = np.nanmean(skews) if len(skews) > 0 else 0
    return mean if np.isfinite(mean) else 0

# We define a function for extracting the standard deviation of all the Skewnesses extracted from all the columns of a matrix
def SkewnessSTD(X):
    skews = Skewnesses(X)
    std = np.nanstd(skews) if len(skews) > 0 else 0
    return std if np.isfinite(std) else 0

# We define a function for extracting the Class Entropy
def ClassEntropy(y):
    labels = 1 if len(y.shape) == 1 else y.shape[1]
    if labels == 1:
        y = y.reshape((-1, 1))

    entropies = []
    for i in range(labels):
        occurence_dict = defaultdict(float)
        for value in y[:, i]:
            occurence_dict[value] += 1
        entropies.append(scipy.stats.entropy([occurence_dict[key] for key in
                                             occurence_dict], base=2))

    return np.mean(entropies)

# We define the function that extracts all the metafeatures for a dataset given in the form of (Features, Class)
def getMetaFeatures(X,Y):
	metafeatures = []

    # Number of instances (rows)
	number_instances = float(X.shape[0])
    # Log Number of instances
	log_number_instances = np.log(number_instances)
    # Number of features (columns)
	number_features = float(X.shape[1])
    # Log Number of features
	log_number_features = np.log(number_features)
    # Proportion of features that are numeric
	proportion_numeric_features = 0.5 ## we hand input this number for each dataset
    # Dataset Ratio: Number of features divided by number of instances
	dataset_ratio = float(number_features)/float(number_instances)
    # Majority class/Class imbalance metric: percentage of the majority class (we assume it's a binary class)
	class_imbalance = max(np.mean(Y.astype(np.int)), 1-np.mean(Y.astype(np.int)))
    # Minimum Kurtosis of all the columns in the dataset
	kurtosis_min = KurtosisMin(X)
    # Maximum Kurtosis of all the columns in the dataset
	kurtosis_max = KurtosisMax(X)
    # Mean Kurtosis of all the columns in the dataset
	kurtosis_mean = KurtosisMean(X)
    # Standard deviation of all the Kurtosisses of all the columns in the dataset
	kurtosis_std = KurtosisSTD(X)
    # Minimum Skewness of all the columns in the dataset
	skewness_min = SkewnessMin(X)
    # Maximum Skewness of all the columns in the dataset
	skewness_max = SkewnessMax(X)
    # Mean Skewness of all the columns in the dataset
	skewness_mean = SkewnessMean(X)
    # Standard deviation of all the Skewnesses of all the columns in the dataset
	skewness_std = SkewnessSTD(X)
    # Class entropy
	class_entropy = ClassEntropy(Y)

	metafeatures.append(number_instances)
	metafeatures.append(log_number_instances)
	metafeatures.append(number_features)
	metafeatures.append(log_number_features)
	metafeatures.append(proportion_numeric_features)
	metafeatures.append(dataset_ratio)
	metafeatures.append(class_imbalance)
	metafeatures.append(kurtosis_min)
	metafeatures.append(kurtosis_max)
	metafeatures.append(kurtosis_mean)
	metafeatures.append(kurtosis_std)
	metafeatures.append(skewness_min)
	metafeatures.append(skewness_max)
	metafeatures.append(skewness_mean)
	metafeatures.append(skewness_std)
	metafeatures.append(class_entropy)

	return metafeatures

# We extract and save on a .csv all the metafeatures for all the datasets

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'w') as csvfile:
    fieldnames = ['number_instances', 'log_number_instances', 'number_features', 'log_number_features',
    'proportion_numeric_features', 'dataset_ratio', 'class_imbalance', 'kurtosis_min', 'kurtosis_max',
    'kurtosis_mean', 'kurtosis_std', 'skewness_min', 'skewness_max', 'skewness_mean', 'skewness_std', 'class_entropy', 'dataset']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

url1 = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
dataset1 = np.loadtxt(urllib.urlopen(url1), delimiter=",")
X = dataset1[:,0:8]
Y = dataset1[:,8]

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 1
    	})

# Dataset 2

url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
dataset2 = np.genfromtxt(urllib.urlopen(url2),dtype=np.str, delimiter=",")

enc_label = LabelEncoder()
dataset2_categ = enc_label.fit_transform(np.array(dataset2[:,1]))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,3]))))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,5]))))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,6]))))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,7]))))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,8]))))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,9]))))
dataset2_categ = np.column_stack((dataset2_categ, enc_label.fit_transform(np.array(dataset2[:,13]))))

enc_onehot = OneHotEncoder()
dataset2_categ = enc_onehot.fit_transform(dataset2_categ)

cols = [str(i) + '_' + str(j) for i in range(0,8) for j in range(0,enc_onehot.n_values_[i]) ]
dataset2_categ_df = pd.DataFrame(dataset2_categ.toarray(),columns=cols)

dataset2_new = dataset2_categ_df
dataset2_new['cont1'] = dataset2[:,0].astype(np.int)
dataset2_new['cont2'] = dataset2[:,2].astype(np.int)
dataset2_new['cont3'] = dataset2[:,4].astype(np.int)
dataset2_new['cont4'] = dataset2[:,10].astype(np.int)
dataset2_new['cont5'] = dataset2[:,11].astype(np.int)
dataset2_new['cont6'] = dataset2[:,12].astype(np.int)
dataset2_new['target'] = dataset2[:,14]

dataset2 = pd.DataFrame(dataset2_new)

dim = dataset2.shape[1] - 1

X = dataset2.ix[:,:dim].as_matrix()
Y = LabelEncoder().fit_transform(dataset2['target'])

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 6/14, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 2
    	})

# Dataset 3

url3 = "http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
dataset3 = np.genfromtxt(urllib.urlopen(url3),dtype=np.str, delimiter=",")

for i in np.arange(0,10):
	dataset3[:,i] = dataset3[:,i].astype(np.float)

X = dataset3[:,0:10]
Y = LabelEncoder().fit_transform(dataset3[:,10])

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 3
    	})

# Dataset 4

url4 = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataset4 = np.loadtxt(urllib.urlopen(url4), delimiter=";", skiprows = 1)
X = dataset4[:,0:11]
Y = dataset4[:,11]
Y[Y<6] = 0
Y[Y>0] = 1

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 4
    	})

# Dataset 5

url5 = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
dataset5 = np.loadtxt(urllib.urlopen(url5), delimiter=",")
X = dataset5[:,0:57]
Y = dataset5[:,57]

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 5
    	})

# Dataset 6

url6_X = "http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data"
X = np.loadtxt(urllib.urlopen(url6_X))
url6_Y = "http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels"
Y = np.loadtxt(urllib.urlopen(url6_Y))

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 6
    	})

# Dataset 7

url7 = "http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data"
dataset7 = np.genfromtxt(urllib.urlopen(url7),dtype=np.str, delimiter=",")

enc_label = LabelEncoder()
dataset7_new = enc_label.fit_transform(np.array(dataset7[:,0]))
for i in np.arange(1,36):
	dataset7_new = np.column_stack((dataset7_new, enc_label.fit_transform(np.array(dataset7[:,i]))))

enc_onehot = OneHotEncoder()
dataset7_new = enc_onehot.fit_transform(dataset7_new)

cols = [str(i) + '_' + str(j) for i in range(0,36) for j in range(0,enc_onehot.n_values_[i]) ]
X = pd.DataFrame(dataset7_new.toarray(),columns=cols).as_matrix()
Y = LabelEncoder().fit_transform(dataset7[:,36])

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 0, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 7
    	})

## Dataset 8

url8 = "http://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
dataset8 = np.genfromtxt(urllib.urlopen(url8),dtype=np.str, delimiter=",")
dataset8[:,0] = dataset8[:,0].astype(np.int)
dataset8[:,1] = dataset8[:,1].astype(np.int)
# dataset8[:,2] = dataset8[:,2].astype(np.float)
for i in np.arange(3,1558):
	dataset8[:,i] = dataset8[:,i].astype(np.int)
X = scipy.delete(dataset8[:,0:1558], 2, 1)
Y = LabelEncoder().fit_transform(dataset8[:,1558])

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 8
    	})

## Dataset 9

url9 = "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
dataset9 = np.loadtxt(urllib.urlopen(url9), delimiter=",")
X = dataset9[:,0:4]
Y = dataset9[:,4]

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 9
    	})

## Dataset 10

dataset10 = pd.read_csv('/Users/Juan-Pablo/Documents/titanic_train.csv' % os.environ)

dataset10 = dataset10.as_matrix()
dataset10 = pd.DataFrame(dataset10)

enc_label = LabelEncoder()
dataset10_new = enc_label.fit_transform(np.array(dataset10[:][4]))
dataset10_new = np.column_stack((dataset10_new, enc_label.fit_transform(np.array(dataset10[:][11]))))

enc_onehot = OneHotEncoder()
dataset10_new = enc_onehot.fit_transform(dataset10_new)

cols = [str(i) + '_' + str(j) for i in range(0,2) for j in range(0,enc_onehot.n_values_[i]) ]
X = pd.DataFrame(dataset10_new.toarray(),columns=cols)

X['cont1'] = dataset10[:][2].astype(np.int)
#X['cont2'] = dataset10[:][5].astype(np.int)
X['cont3'] = dataset10[:][6].astype(np.int)
X['cont4'] = dataset10[:][7].astype(np.int)
X['cont5'] = dataset10[:][9].astype(np.float)
X = X.as_matrix()
Y = dataset10[:][1].astype(np.int)

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 4/6, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 10
    	})

## Dataset 11

url11 = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt"
dataset11 = np.genfromtxt(urllib.urlopen(url11))

X = dataset11[:,0:85]
Y = dataset11[:,85]

sm = SMOTEENN()
X, Y = sm.fit_sample(X, Y)

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 11
    	})

## Dataset 12

url12 = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data"
dataset12 = np.loadtxt(urllib.urlopen(url12), dtype=np.str, delimiter=",", skiprows = 1)

for i in np.arange(1,10):
	if i == 1 or i == 6:
		dataset12[:,i] = dataset12[:,i].astype(np.int)
	elif i == 5:
		continue
	else:
		dataset12[:,i] = dataset12[:,i].astype(np.float)

X1 = dataset12[:,1:5]
X2 = dataset12[:,6:10]
X = np.concatenate((X1, X2), axis=1)
Y = dataset12[:,10]

metaFeatures = getMetaFeatures(X,Y)

with open('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)

    writer.writerow({'number_instances': metaFeatures[0], 'log_number_instances': metaFeatures[1], 
    	'number_features' : metaFeatures[2], 'log_number_features' : metaFeatures[3],
    	'proportion_numeric_features': 1, 'dataset_ratio' : metaFeatures[5],
    	'class_imbalance': metaFeatures[6], 'kurtosis_min' : metaFeatures[7], 'kurtosis_max' : metaFeatures[8],
    	'kurtosis_mean' : metaFeatures[9], 'kurtosis_std' : metaFeatures[10], 'skewness_min' : metaFeatures[11],
    	'skewness_max' : metaFeatures[12], 'skewness_mean' : metaFeatures[13], 'skewness_std' : metaFeatures[14],
    	'class_entropy' : metaFeatures[15], 'dataset' : 12
    	})
