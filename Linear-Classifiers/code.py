from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import math
import csv
NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test) 


def train(X_train, y_train, reg=1):
    ''' Build a model from X_train -> y_train '''
    X=np.dot(X_train.T,X_train)
    XY=np.dot(X_train.T,y_train)
    return np.linalg.solve(X + np.dot(reg,np.identity(len(X))),XY)

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    XTX=np.dot(X_train.T,X_train)
    XY=np.dot(X_train.T,y_train)
    #m=X_train.shape[1]
    n=X_train.shape[0]
    lambdI=reg*np.identity(len(XTX))
    XTXI=XTX +lambdI
    #W=np.zeros((m,p))
    W=XY
    for i in  range(num_iter):
        W= W -(alpha/n)*XTXI.dot(W) +(alpha/n)*XY

    return W

def shuffle(X,Y):
    '''  Shuffles X and Y at the same time '''
    rd=np.random.permutation(Y.shape[0])
    return X[rd], Y[rd]

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    
    #W=np.zeros((X_train.shape[1],10))
    
    lambdI=reg*np.identity(X_train.shape[1])
    X, Y = shuffle(X_train, y_train)
    XTXI=np.dot(X.T,X) + lambdI
    XY=X.T.dot(Y),add
    W=np.zeros(XY.shape)
    avg=X_train.shape[0]
    n=np.random.randint(W.shape[0], size=num_iter)
    p=np.random.randint(W.shape[1], size=num_iter)
    for i in range(num_iter):
        ns=n[i]
        ps=p[i]
        W[n,p]=W[n,p]-(alpha/avg)*XTXI.dot(W)[n,p] + (alpha/avg)*XY[n,p]

    return W

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    L_train=np.zeros((X_train.shape[0], NUM_CLASSES))
    k=0
    for i in labels_train:
        L_train[k,i]=1
        k+=1
    return L_train

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    CP=np.dot(X,model)
    
    return np.argmax(CP,axis=1)

def phi(X):
    ''' Featurize the inputs using random Fourier features '''
    # return W^TX +B (first we transpose X to get the pictures (features) then we transposed back to add B 
    #dimension n X p
    return math.sqrt(2)*np.cos(np.dot(X,W) + b)
    
def forkaggle(K):
    with open('datak8.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Id','Category'])
        k=0
        for i in K:
            spamwriter.writerow([k, i])
            k+=1


if __name__ == "__main__":
    sigma=0.23
    p=1250 #parameter that will will tweak
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    b=np.random.uniform(0,2*np.pi,p) #Constant b in terms of matrix nXp
    W=np.random.normal(0,sigma,(X_train.shape[1],p)) # already in tems of pXd no need to transpose
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test, = phi(X_train), phi(X_test)
    print ('go phi')

    #model = train(X_train, y_train, reg=0.1)

    #pred_labels_train = predict(model, X_train)
    #pred_labels_test = predict(model, X_test)  
    #print("Closed form solution")
    #print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    #print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
    #pred_labels_ktest = pred_labels_test.astype(int)
    #forkaggle(pred_labels_ktest)

    model = train_gd(X_train, y_train, alpha=7e-6, reg=0.1, num_iter=16000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    #model = train_sgd(X_train, y_train, alpha=2.5e-3, reg=0.1, num_iter=12000)
    #pred_labels_train = predict(model, X_train)
    #pred_labels_test = predict(model, X_test)
    
    #print("Stochastic gradient descent")
    #print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    #print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
