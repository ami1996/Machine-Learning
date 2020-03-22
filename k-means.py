# -*- coding: utf-8 -*-
"""
@author: Amit kumar
"""

import numpy as np
import pandas as pd
from copy import deepcopy

def kmeans(X,k):
    # setting initial centroids
    C = []
    c = np.random.randint(0,len(X),size=k)
    for i in c:
        C.append(list(X[i]))
    C = np.array(C,dtype = np.float32)
    print("\nInitial Centroids")
    print(C)
    
    # Euclidean distance calculation
    def distance(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)
    
    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = distance(C, C_old, None)
    # Loop will run till the error becomes zero
    iter_count = 0
    while error != 0:
        iter_count += 1
        print("\nIteration" + str(iter_count))
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = distance(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
            print("\ncluster"+str(i))
            print(np.array(points))
        print("\nnew centroid: ")
        print(C)
        error = distance(C, C_old, None)
        
if __name__ == "__main__":
    # Importing the dataset
    url = "iris_dataset\iris.data"
    names = ['sepal-length','sepal-width','petal-length','petal-width','class']
    data = pd.read_csv(url,names = names)
    print("Input Data and Shape")
    print(data.shape)
    # adjusting dataset 
    array = data.values
    f1 = array[:,0]
    f2 = array[:,1]
    f3 = array[:,2]
    f4 = array[:,3]
    X = np.array(list(zip(f1, f2, f3, f4)))
    
    k = int(input("number of clusters: "))
    kmeans(X,k)
