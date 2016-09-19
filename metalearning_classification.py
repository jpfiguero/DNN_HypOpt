from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
import scipy

## We load the Metafeatures of all the datasets

datasets = pd.read_csv('/Users/Juan-Pablo/Documents/metalearning/metalearning.csv' % os.environ)
datasets = datasets.as_matrix()
nrows = pd.DataFrame(datasets).shape[0]
ncols = pd.DataFrame(datasets).shape[1]
X = datasets[:,0:ncols-1]
Y = datasets[:,ncols-1]

neigh = KNeighborsClassifier(n_neighbors=1)

## Dataset 1

X1 = X[1:nrows-1,:]
Y1 = Y[1:nrows-1]

neigh.fit(X1, Y1) 

print("Closes dataset to Dataset #1 is Dataset #%d" % neigh.predict(X[0,:]))

# Closest dataset: 10


## Dataset 2

X2 = scipy.delete(X, 1, 0)
Y2 = scipy.delete(Y, 1, 0)

neigh.fit(X2, Y2) 

print("Closes dataset to Dataset #2 is Dataset #%d" % neigh.predict(X[1,:]))

# Closest dataset: 3


## Dataset 3

X3 = scipy.delete(X, 2, 0)
Y3 = scipy.delete(Y, 2, 0)

neigh.fit(X3, Y3) 

print("Closes dataset to Dataset #3 is Dataset #%d" % neigh.predict(X[2,:]))

# Closest dataset: 11


## Dataset 4

X4 = scipy.delete(X, 3, 0)
Y4 = scipy.delete(Y, 3, 0)

neigh.fit(X4, Y4) 

print("Closes dataset to Dataset #4 is Dataset #%d" % neigh.predict(X[3,:]))

# Closest dataset: 9


## Dataset 5

X5 = scipy.delete(X, 4, 0)
Y5 = scipy.delete(Y, 4, 0)

X5 = scipy.delete(X5, 6, 0)
Y5 = scipy.delete(Y5, 6, 0)

neigh.fit(X5, Y5) 

print("Closes dataset to Dataset #5 is Dataset #%d" % neigh.predict(X[4,:]))

# Closest dataset: 8, and then 7


## Dataset 6

X6 = scipy.delete(X, 5, 0)
Y6 = scipy.delete(Y, 5, 0)

neigh.fit(X6, Y6) 

print("Closes dataset to Dataset #6 is Dataset #%d" % neigh.predict(X[5,:]))

# Closest dataset: 4


## Dataset 7

X7 = scipy.delete(X, 6, 0)
Y7 = scipy.delete(Y, 6, 0)

neigh.fit(X7, Y7) 

print("Closes dataset to Dataset #7 is Dataset #%d" % neigh.predict(X[6,:]))

# Closest dataset: 5


## Dataset 8

X8 = scipy.delete(X, 7, 0)
Y8 = scipy.delete(Y, 7, 0)

X8 = scipy.delete(X8, 5, 0)
Y8 = scipy.delete(Y8, 5, 0)

neigh.fit(X8, Y8) 

print("Closes dataset to Dataset #8 is Dataset #%d" % neigh.predict(X[7,:]))

# Closest dataset: 6, and then 5


## Dataset 9

X9 = scipy.delete(X, 8, 0)
Y9 = scipy.delete(Y, 8, 0)

neigh.fit(X9, Y9) 

print("Closes dataset to Dataset #9 is Dataset #%d" % neigh.predict(X[8,:]))

# Closest dataset: 4


## Dataset 10

X10 = scipy.delete(X, 9, 0)
Y10 = scipy.delete(Y, 9, 0)

neigh.fit(X10, Y10) 

print("Closes dataset to Dataset #10 is Dataset #%d" % neigh.predict(X[9,:]))

# Closest dataset: 1


## Dataset 11

X11 = scipy.delete(X, 10, 0)
Y11 = scipy.delete(Y, 10, 0)

neigh.fit(X11, Y11) 

print("Closes dataset to Dataset #11 is Dataset #%d" % neigh.predict(X[10,:]))

# Closest dataset: 5


## Dataset 12

X12 = scipy.delete(X, 11, 0)
Y12 = scipy.delete(Y, 11, 0)

neigh.fit(X12, Y12) 

print("Closes dataset to Dataset #12 is Dataset #%d" % neigh.predict(X[11,:]))

# Closest dataset: 1
