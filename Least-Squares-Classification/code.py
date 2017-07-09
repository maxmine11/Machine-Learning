from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0.0000005):
    ''' Build a model from X_train -> y_train '''
    y_labels=one_hot(y_train)
    X=np.dot(X_train.T,X_train)
    XY=np.dot(X_train.T,y_labels)
    #X=np.zeros((X_train.shape[1], X_train.shape[1]))
    #XY=np.zeros((X_train.shape[1],NUM_CLASSES))
    #for i in range(X_train.shape[1]):
      # X+=np.outer(X_train[:,i],X_train[:,i])
       #XY+=np.outer(X_train[:,i],y_labels[:,i])
    #X_f=np.linalg.inv((X + np.dot(reg,np.identity(len(X)))))
    return np.linalg.solve(X + np.dot(reg,np.identity(len(X))),XY)
    #return np.zeros((X_train.shape[0], y_train.shape[0]))

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    L_train=np.zeros(( X_train.shape[0], NUM_CLASSES))
    k=0
    for i in labels_train:
        L_train[k][i]=1
        k+=1
    return L_train
def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    CP=np.dot(X,model)
    predict_max=np.argmax(CP,axis=1)
    return predict_max

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    model = train(X_train, labels_train)
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    #print(X_train[1,:])
    #print(model[:,1])
    #print (np.dot(X_train,model)[0:5,:])
    print(len(X_train[0]))
    print(labels_train[0])
    print(labels_train.shape)
    print (y_train[0:6,:])
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
