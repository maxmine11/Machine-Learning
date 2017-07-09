import sklearn.metrics as metrics
import numpy as np
import scipy.io as sio
import math
import csv
import matplotlib.pyplot as plt


def load_dataset():
    datamat=sio.loadmat('./data/spam.mat')
    X_train=datamat['Xtrain']
    X_test=datamat['Xtest']
    y_train=datamat['ytrain']  
    return X_train, X_test, y_train

def standarize(Xtrain, Xtest):
    Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0) # Standarize X_train, unit_var      
    Xtest = (Xtest - Xtest.mean(axis=0)) / Xtest.std(axis=0) #Standarize X_test,                     
    return Xtrain, Xtest

def log_transform(X, Xtest):
    X=np.log(X + 0.1)
    Xtest=np.log(Xtest + 0.1)
    return X, Xtest

def binarize(X, Xtest):
    X=np.where(X>0, 1, 0)
    Xtest=np.where(Xtest>0, 1, 0)
    return X, Xtest
#Proble 4.1 Gradient Descent
def train_gd(X, y_train, alpha=0.1, reg=0, num_iter=10000):
    '''Build a model from X_train -> y_train using batch gradient descent for logistic regression'''
    XTY=np.dot(X.T,y_train)
    n=X.shape[0]
    lambd=reg*np.identity(len(XTY))
    W=np.ones((X.shape[1],1),dtype=float)
    for i in range(num_iter):
        u=1.0/(1.0+np.exp(-np.dot(X,W)))
        W=W-(alpha/n)*(2*lambd.dot(W)-XTY+X.T.dot(u))
    print(u.shape)
    return W

def shuffle(X,Y):
    '''  Shuffles X and Y at the same time '''
    rd=np.random.permutation(Y.shape[0])
    return X[rd], Y[rd]

def sigmoidi(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = math.exp(x)
        return z / (1.0 + z)
#Problem 4.2 Stochastic gradient descent
def train_sgd(X, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient for logistic regression withl2 regularization.'''
    n=X.shape[0]
    X, Y = shuffle(X, y_train)
    W=np.zeros((X.shape[1],1),dtype=float)[:,0]
    select=np.random.randint(W.shape[0], size=num_iter)
    for i in range(num_iter):
        k=select[i]
        u=sigmoidi(np.dot(X.T[:,k],W))
        W=W-(alpha/n)*(2*reg*W-(Y[k]-u)*X.T[:,k])
    return W

# Problem 4.3, the lerning rate alpha is proportional to 1/t, it decreases with the number of iterations
def train_sgdn(X, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient for logistic regression withl2 regularization.'''
    n=X.shape[0]
    X, Y = shuffle(X, y_train)
    W=np.zeros((X.shape[1],1),dtype=float)[:,0]
    
    select=np.random.randint(W.shape[0], size=num_iter)
    for i in range(num_iter):
        k=select[i]
        u=sigmoidi(np.dot(X.T[:,k],W))
        W=W-(1/(i+1))*(alpha/n)*(2*reg*W-(Y[k]-u)*X.T[:,k])
    return W


def predict(model, X):
    print( "U shape")
    u=1.0/(1.0+np.exp(-X.dot(model)))
    u=np.array(np.where(u>0.5, 1, 0),dtype=float)
    print(u)
    return u

def forkaggle(K):
    with open('datak.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Id','Category'])
        k=1
        for i in K:
            spamwriter.writerow([k, i[0]])
            k+=1

    
if __name__ == "__main__":
    X_train, X_test, y_train=load_dataset()
    #X_train,X_test=standarize(X_train,X_test)
    X_train,X_test=log_transform(X_train,X_test)
    #X_train,X_test=binarize(X_train,X_test)


    model = train_gd(X_train, y_train, alpha=0.15, reg=0.17, num_iter=60000)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_labels_train)))
    pred_labels_ktest = pred_labels_test.astype(int)
    forkaggle(pred_labels_ktest)
    
    


    model = train_sgd(X_train, y_train, alpha=0.08, reg=0.1, num_iter=300000)                           pred_labels_train = predict(model, X_train)                                                         #pred_labels_test = predict(model, X_test)                                                                                                                                                          
    print("Stochastic gradient descent")                                                                                                                                                                
    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_labels_train)))                                                                                                           
    
    
    model = train_sgdn(X_train, y_train, alpha=0.08, reg=0.1, num_iter=300000)                  
    pred_labels_train = predict(model, X_train)                                                                                                                                                         
    #pred_labels_test = predict(model, X_test)                                                                                                                                                          
    print("Stochastic gradient descent for Learning rate decreasing with 1/t")                                                                                                                              print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_labels_train)))                                                                                                           
    


######################################################################################
#Code for plots of Problem 4

#Stable sigmoid
def sigmoid(x):
    "Numerically-stable sigmoid function."
    xn=np.ones((x.shape[0],1),dtype=float)
    k=0
    
    for i in x:
        if i >= 0:
            z = math.exp(-i)
            xn[k]= 1.0 / (1.0 + z)
            k+=1
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = math.exp(i)
            xn[k]= z / (1.0 + z)
            k+=1
    return xn
def l(X,Y,W,reg):
    u=sigmoid(X.dot(W))
    return reg*W.T.dot(W)-Y.T.dot(np.log(1e-300+u))-(1.0-Y).T.dot(np.log(1.0-u))
#Gradient descent  that creates values for plot of 4.1
def train_gd2(X, y_train, alpha, reg, num_iter):
    '''Build a model from X_train -> y_train using batch gradient descent for logistic regression'''
    XTY=np.dot(X.T,y_train)
    n=X.shape[0]
    LN=np.zeros((num_iter,2))
    LN[:,0]=np.arange(1,num_iter+1)
    lambd=reg*np.identity(len(XTY))
    W=np.zeros((X.shape[1],1),dtype=float)
    for i in range(num_iter):
        u=sigmoid(X.dot(W))
        LN[i,1]=l(X,y_train,W,reg)
        W=W-(alpha/n)*(2*lambd.dot(W)-XTY+X.T.dot(u))
        
    return LN

#Stochastic gradient descent that creates values for plot of 4.2
def train_sgd2(X, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient for logistic regression withl2 regularization.'''
    n=X.shape[0]
    X, Y = shuffle(X, y_train)
    LN=np.zeros((num_iter,2))
    LN[:,0]=np.arange(1,num_iter+1)
    W=np.zeros((X.shape[1],1),dtype=float)[:,0]
    select=np.random.randint(W.shape[0], size=num_iter)
    for i in range(num_iter):
        k=select[i]
        u=sigmoidi(np.dot(X.T[:,k],W))
        LN[i,1]=l(X,y_train,W,reg)
        W=W-(alpha/n)*(2*reg*W-(Y[k]-u)*X.T[:,k])
    return LN
# Stochastic gradient descent with decreasing learning rate that creates valyes for plot of 4.3
def train_sgd2n(X, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient for logistic regression withl2 regularization.'''
    n=X.shape[0]
    X, Y = shuffle(X, y_train)
    LN=np.zeros((num_iter,2))
    LN[:,0]=np.arange(1,num_iter+1)
    W=np.zeros((X.shape[1],1),dtype=float)[:,0]
    select=np.random.randint(W.shape[0], size=num_iter)
    for i in range(num_iter):
        k=select[i]
        u=sigmoidi(np.dot(X.T[:,k],W))
        LN[i,1]=l(X,y_train,W,reg)
        W=W-(1/(i+1))*(alpha/n)*(2*reg*W-(Y[k]-u)*X.T[:,k])
    return LN


######################################################################################
#Graphs for problem 2
X_data=np.array([4,5,5.6,6.8,7,7.2,8,0.8,1,1.2,2.5,2.6,3,4.3],)
Y_data=np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0],dtype=float)
B_in=np.array([1,0],dtype=float)
X_two=np.ones(14)
sigma=X_data.std()
X_std =(X_data -X_data.mean()*X_two)/sigma
X_r=np.vstack((X_std,X_two)).T

def J_update(X,Y,B,reg,num_iter):
    XTX=X.T.dot(X)
    XTY=X.T.dot(Y)
    l=reg*np.identity(len(XTX))
    for i in range(num_iter):
        B=B-np.linalg.inv(2*l+ XTX).dot(2*l.dot(B)-XTY+XTX.dot(B))
    return B

def l_update(X,Y,B,reg,num_iter):
    u=1.0/(1.0+np.exp(-X.dot(B)))
    W = np.zeros((u.shape[0],u.shape[0]), float)
    XTY=X.T.dot(Y)
    l=reg*np.identity(X.shape[1])
    for i in range(num_iter):
        u=1.0/(1.0+np.exp(-X.dot(B)))
        np.fill_diagonal(W, np.multiply(u,(np.ones(len(u))-u)))
        XTWX=X.T.dot(W.dot(X))
        B= B - np.linalg.inv(2*l + XTWX).dot(2*l.dot(B) - XTY +X.T.dot(u))
    return B

B_up=J_update(X_r,Y_data,B_in,0.07,1)
B_upl=l_update(X_r,Y_data,B_in,0.07,3)
plt.plot(X_std,Y_data,'ro')
plt.plot(np.sort(X_std),np.sort(X_r.dot(B_up)),linewidth=2.5,label=r'$Ridge \quad \beta=[0.42,0.50]$')
plt.plot(np.sort(X_std),np.sort(1/(1+np.exp(-X_r.dot(B_upl)))),linewidth=2.5, label=r'$Log \quad \beta=[3.2,0.08]$')
plt.legend(loc='lower right')
plt.suptitle('Logistic regression Vs Ridge regression', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True)
plt.show()

X_nd=np.append(X_std,[3])
X_twod=np.ones(15)
Y_nd=np.append(Y_data,[1])
X_d=np.vstack((X_nd,X_twod)).T

B_upd=J_update(X_d,Y_nd,B_in,0.07,3)
B_upld=l_update(X_d,Y_nd,B_in,0.07,3)
plt.plot(X_nd,Y_nd,'ro')
plt.plot(np.sort(X_nd),np.sort(X_d.dot(B_upd)),linewidth=2,label=r'$Ridge \quad \beta=[0.33,0.46]$')
plt.plot(np.sort(X_nd),np.sort(1/(1+np.exp(-X_d.dot(B_upld)))),linewidth=2, label=r'$Log \quad \beta=[3.2,0.08]$')
plt.legend(loc='lower right')
plt.suptitle('Log Vs Ridge (Additional Data)', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True)
plt.show()

