import numpy as np
import pandas as pd
import partitioning
import confusion

# Euclidean distance calculation
def distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def knn(X_train,X_test,Y_train,Y_test):
    # value of k
    k = int(input("enter the number of neighbours: "))
    # list to store the predicted output by the model
    Y_predicted = []
    
    for i in X_test:
        # calculating distance matrix for each test sample
        distances = distance(np.array(i, dtype = np.float32),np.array(X_train, dtype = np.float32))
        x = list(zip(X_train,Y_train,distances))
        # sorting points on the basis of distance
        sorted_x = sorted(x, key=lambda kv: kv[2])
        #print(sorted_x)
        
        # stores the count for each label
        lcount = list(0 for _ in range(len(labels)))
        for j in range(k):
            for _ in range(len(labels)):
                if sorted_x[j][1] == labels[_]:
                    lcount[_] += 1
        
        # case i : selecting the label having max count
        if(lcount.count(max(lcount)) == 1):
            Y_predicted.append(labels[lcount.index(max(lcount))])
            
        # case ii: more thsn one label have same count
        else:       
            ind = np.where(np.array(lcount) == max(lcount))[0]
            sum = list(0 for _ in range(len(ind)))
            for j in range(k):
                for m in range(len(ind)):
                    if sorted_x[j][1] == labels[ind[m]]:
                        sum[m] += sorted_x[j][2]
            Y_predicted.append(labels[ind[sum.index(min(sum))]])
            
    #print(list(zip(Y_test,Y_predicted,Y_test == Y_predicted)))
    return Y_predicted

if __name__ == "__main__":
    # Importing the dataset
    url = "iris_dataset\iris.data"
    names = ['sepal-length','sepal-width','petal-length','petal-width','class']
    labels = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    data = pd.read_csv(url,names = names)
    array = data.values
    X = array[:,0:-1]
    Y = array[:,-1]
        
    # ratio of test data
    r = float(input("enter train_test ratio: "))
    # stratified splitting
    X_train,X_test,Y_train,Y_test = partitioning.train_test_stratsplit(X,Y,r)
    
    Y_predicted = knn(X_train,X_test,Y_train,Y_test)
    
    print("confusion matrix:") 
    mat = confusion.confusion_matrix(Y_test,Y_predicted,labels)
    print(mat)
        
    confusion.perf_measures(mat,labels)
    
