# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:19:43 2020

@author: Amit kumar
"""

import random
import numpy as np
from pandas import read_csv

def train_test_split(X,Y,r):
    c = random.sample(range(len(X)),k = int(r*len(X)))

    X_tr = [];X_te = []
    Y_tr = [];Y_te = []
    for i in range(len(X)):
        if i in c:
            X_tr.append(X[i])
            Y_tr.append(Y[i])
        else:
            X_te.append(X[i])
            Y_te.append(Y[i])
            
    X_train = np.array(X_tr, dtype = int)
    X_test = np.array(X_te, dtype = int)
    Y_train = np.array(Y_tr, dtype = object)
    Y_test = np.array(Y_te, dtype = object)
    
    #print(X_train,Y_train)
    #print(X_test,Y_test)
    return(X_train,X_test,Y_train,Y_test)
    
def train_test_stratsplit(X,Y,r):
    X_tr = [];X_te = []
    Y_tr = [];Y_te = []
    
    c1 = random.sample(range(int((1/3)*len(X))),k = int(r*((1/3)*len(X))))
    c2 = random.sample(range(int((1/3)*len(X)),int((2/3)*len(X))),k = int(r*((1/3)*len(X))))
    c3 = random.sample(range(int((2/3)*len(X)),len(X)),k = int(r*((1/3)*len(X))))
    
    c = c1+c2+c3
    
    for i in range(len(X)):
        if i in c:
            X_tr.append(X[i])
            Y_tr.append(Y[i])
        else:
            X_te.append(X[i])
            Y_te.append(Y[i])
            
    X_train = np.array(X_tr, dtype = int)
    X_test = np.array(X_te, dtype = int)
    Y_train = np.array(Y_tr, dtype = object)
    Y_test = np.array(Y_te, dtype = object)
    
    #print(X_train,Y_train)
    #print(X_test,Y_test)
    return(X_train,X_test,Y_train,Y_test)
    
def train_test_valid_stratsplit(X,Y,r):
    X_tr = [];X_te = [];X_va = []
    Y_tr = [];Y_te = [];Y_va = []
    
    c1 = random.sample(range(int((1/3)*len(X))),k = int(r*((1/3)*len(X))))
    c2 = random.sample(range(int((1/3)*len(X)),int((2/3)*len(X))),k = int(r*((1/3)*len(X))))
    c3 = random.sample(range(int((2/3)*len(X)),len(X)),k = int(r*((1/3)*len(X))))
    c4 = random.sample(c1,k = int((1-r)*(r*((1/3)*len(X)))))
    c5 = random.sample(c2,k = int((1-r)*(r*((1/3)*len(X)))))
    c6 = random.sample(c3,k = int((1-r)*(r*((1/3)*len(X)))))
    
    c = c1+c2+c3
    cv = c4+c5+c6
    
    for i in range(len(X)):
        if i in c:
            if i not in cv:
                X_tr.append(X[i])
                Y_tr.append(Y[i])
            else:
                X_va.append(X[i])
                Y_va.append(Y[i])
        else:
            X_te.append(X[i])
            Y_te.append(Y[i])
            
    X_train = np.array(X_tr, dtype = int)
    X_validation = np.array(X_va, dtype = int)
    X_test = np.array(X_te, dtype = int)
    Y_train = np.array(Y_tr, dtype = object)
    Y_validation = np.array(Y_va, dtype = object)
    Y_test = np.array(Y_te, dtype = object)
    
    #print(X_train,Y_train)
    #print(X_validation,Y_validation)
    #print(X_test,Y_test)
    return(X_train,X_validation,X_test,Y_train,Y_validation,Y_test)

if __name__ == "__main__":
    url = "iris_dataset\iris.data"
    names = ['sepal-length','sepal-width','petal-length','petal-width','class']
    dataset = read_csv(url,names = names)
    
    array = dataset.values
    X = array[:,0:-1]
    Y = array[:,-1]
    
    print("1:train_test\n2:train_test_strat\n3:train_test_valid")
    choice = int(input("choice: "))
    
    r = float(input("ratio: "))
    
    if(choice == 1):
        train_test_split(X,Y,r)
    elif(choice == 2):
        train_test_stratsplit(X,Y,r)
    elif(choice == 3):
        train_test_valid_stratsplit(X,Y,r)
    else:
        print("wrong choice")
        
        