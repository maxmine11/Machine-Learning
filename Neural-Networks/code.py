import sklearn.metrics as metrics
from mnist import MNIST
import numpy as np
import scipy.io as sio
import math
import csv
import time
"""                                                                                                                                                                                                                                                                             
Change this code however you want.                                                                                                                                                                                                                                              
"""
NUM_CLASSES = 10

def load_dataset():
    ''' Load data from our data folder in hw4 repository'''
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, _ = map(np.array, mndata.load_testing())
    X_test=X_test/255.0
    X_train=X_train/255.0
    X_train,X_validation =X_train[:-10000,:],X_train[X_train.shape[0]-10000:,:]
    labels_train,labels_validation=labels_train[:-10000],labels_train[labels_train.shape[0]-10000:]
    #X_test=X_test/255.0
    # Remember to center and normalize the data...                                                                                                                                                                                                                              
    return X_train, labels_train, X_validation, labels_validation, X_test


def standarize(Xtrain, Xtest):
    ''' Centers and normalizes all features '''
    Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0)                                                
    Xtest = (Xtest - Xtest.mean(axis=0)) / Xtest.std(axis=0)                                                                                                                                        
    return Xtrain, Xtest




def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    L_train=np.zeros(( X_train.shape[0], NUM_CLASSES))
    k=0
    for i in labels_train:
        L_train[k][i]=1
        k+=1
    return L_train


def shuffle(X,Y):
    '''  Shuffles X and Y at the same time '''
    rd=np.random.permutation(Y.shape[0])
    return X[rd], Y[rd]

def softmax(x):
    ''' Soft max activation function, similar to sigmoid but for the multiple case class scenario for output neurons'''
    if x.shape>(x.shape[0],1):
        sum_s=np.diag(1./np.sum(np.exp(x),axis=0)) #adds together entries on each column separately, then takes the inverse of each entry in the array and maps it to a diagonal matrix. Dimension nxn.
        z=np.exp(x) # Takes the exponetial of each entry of the array x
        return np.dot(z,sum_s)
    else:
        s=np.sum(np.exp(x))
        return np.exp(x)/s
    
def trainNeuralNetwork(X,Y,V,W,eta_1=0.01,eta_2=0.01,b_size=50,b=1.0,n_iter=150000):
    ''' Train the neural network  by triaining W and V with stochastic gradient descent. The parameter eta represents the learning rate,
    b_size represents the batch size, b represents the bias term, Y contains all labels, and X contains all the samples'''
    startTime = time.time()
    X,Y=shuffle(X,Y)
    b_s=b_size
    bias_in=b*np.ones(X.shape[0])
    bias_hid=np.append(bias_in,b)
    XT=np.vstack((X.T,bias_in))
    for i in range(n_iter):
        index=np.random.randint(XT.shape[1])
        XT1=XT[:,index]
        #S_h=np.vstack((V.dot(XT),bias_hid))
        S_h=np.append(V.dot(XT1),1.0)
        H=S_h*(S_h>0)
        H=H.reshape((H.shape[0],1))
        delta_L=softmax(np.dot(W,H)) -Y.T[:,index].reshape((10,1))
        delta_h=np.dot(W.T,delta_L)*(S_h.reshape((201,1))>0) #Needs to account for ReUL prime activation function
        grad_W=np.dot(delta_L,H.T)
        grad_V=np.dot(delta_h[:-1],XT1.reshape((1,XT1.shape[0])))
        W=W-eta_1*grad_W
        V=V-eta_2*grad_V
    print "The printing time for %s iterations is:" % n_iter
    print "%s seconds" %(time.time() - startTime)
    return W,V
def predictNeuralNetwork(X,W_trained,V_trained):
    b=1.0
    bias_in=b*np.ones(X.shape[0])
    #bias_hid=np.append(bias_in,b)
    XT=np.vstack((X.T,bias_in))
    S_h=np.vstack((V_trained.dot(XT),bias_in))
    H=S_h*(S_h>0)
    Z=softmax(np.dot(W_trained,H))
    predict_max=np.argmax(Z,axis=0)
    return predict_max
    
def forkaggle(K):
    with open('datak2.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Id','Category'])
        k=0
        for i in K:
            spamwriter.writerow([k, i])
            k+=1


if __name__ == "__main__":
    X_train, labels_train, X_validation, labels_validation, X_test = load_dataset()
    y_validation=one_hot(labels_validation)                                              
    y_train = one_hot(labels_train)
    sigma=0.5
    n_input= X_train.shape[1]
    n_hidden=200
    n_out=10
    V=np.random.normal(0,sigma,(n_hidden,n_input + 1))
    W=np.random.normal(0,sigma,(n_out, n_hidden +1))
    W_n,V_n= trainNeuralNetwork(X_train, y_train,V,W,eta_1=0.002,eta_2=0.002,b_size=50,b=1.0,n_iter=250000)
    pred_labels_train = predictNeuralNetwork(X_train,W_n,V_n)        
    pred_labels_validation = predictNeuralNetwork(X_validation,W_n,V_n)
    pred_labels_test=predictNeuralNetwork(X_test,W_n, V_n)
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_validation, pred_labels_validation)))
    pred_labels_ktest = pred_labels_test.astype(int)                                                                                                                                                       
    forkaggle(pred_labels_ktest)

