#from pandas import read_csv
import numpy as np
        
def confusion_matrix(Y_test,Y_predicted,labels):
    mat = np.zeros((len(labels),len(labels)))

    for _ in range(len(Y_predicted)):
        for i in range(len(labels)):
            for j in range(len(labels)):
                if(Y_predicted[_] == labels[i] and Y_test[_] == labels[j]):
                    mat[i][j] += 1
    return mat
    
def perf_measures(mat,labels):
    print("accuracy : "+str(np.trace(mat)/np.sum(mat)))
    
    for i in range(len(labels)):
        print("recall_"+labels[i]+" : "+str(mat[i][i]/np.sum(mat,axis=1)[i]))
    
    for i in range(len(labels)):
        print("precision_"+labels[i]+" : "+str(mat[i][i]/np.sum(mat,axis=0)[i]))   
    
    for i in range(len(labels)):
        print("f1score_"+labels[i]+" : "+str(
              2*(mat[i][i]/np.sum(mat,axis=0)[i] * mat[i][i]/np.sum(mat,axis=1)[i])/
              (mat[i][i]/np.sum(mat,axis=0)[i] + mat[i][i]/np.sum(mat,axis=1)[i])))
        
if __name__ == "__main__":
    '''url = "iris_dataset\iris.data"
    names = ['sepal-length','sepal-width','petal-length','petal-width','class']
    dataset = read_csv(url,names = names)'''   
    labels = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    
    Y_predicted = ['Iris-setosa','Iris-setosa','Iris-setosa','Iris-setosa',
                   'Iris-setosa','Iris-setosa','Iris-setosa','Iris-setosa',
                   'Iris-setosa','Iris-setosa','Iris-versicolor','Iris-versicolor',
                   'Iris-versicolor','Iris-versicolor','Iris-versicolor',
                   'Iris-versicolor','Iris-virginica','Iris-versicolor',
                   'Iris-versicolor','Iris-versicolor','Iris-virginica',
                   'Iris-virginica','Iris-virginica','Iris-virginica',
                   'Iris-virginica','Iris-virginica','Iris-virginica',
                   'Iris-virginica','Iris-virginica','Iris-virginica']
    
    Y_test = ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
           'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
           'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica',
           'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
           'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
           'Iris-virginica', 'Iris-virginica', 'Iris-virginica']      
     
    print("confusion matrix:") 
    mat = confusion_matrix(Y_test,Y_predicted,labels)
    print(mat)
    
    perf_measures(mat,labels)
