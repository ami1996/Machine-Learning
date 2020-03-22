# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:38:05 2020

@author: Amit kumar
"""

import numpy as np
from sklearn import datasets

def cov(x, y = None):
    if y == None:
        y = x
        
    return np.matmul((x - x.mean(axis = 1).reshape(x.shape[0],1)),
                     (y - y.mean(axis = 1).reshape(x.shape[0],1)).transpose()
                     )/ (x.shape[1]-1)

if __name__ == "__main__" :
    X = datasets.load_iris().data
    print(cov(X))