
# coding: utf-8

# In[1]:



import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:

X_data=np.array([4,5,5.6,6.8,7,7.2,8,0.8,1,1.2,2.5,2.6,3,4.3],)


# In[3]:

Y_data=np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0],dtype=float)
B_in=np.array([1,0],dtype=float)
X_two=np.ones(14)
sigma=X_data.std()
X_std =(X_data -X_data.mean()*X_two)/sigma
X_r=np.vstack((X_std,X_two)).T
print (X_std)


# In[5]:

def J_update(X,Y,B,reg,num_iter):
    XTX=X.T.dot(X)
    XTY=X.T.dot(Y)
    l=reg*np.identity(len(XTX))
    for i in range(num_iter):
        B=B-np.linalg.inv(2*l+ XTX).dot(2*l.dot(B)-XTY+XTX.dot(B))
    return B
    


# In[66]:


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



# In[16]:

B_up=J_update(X_r,Y_data,B_in,0.07,1)
print(B_up)


# In[20]:

B_upl=l_update(X_r,Y_data,B_in,0.07,3)
print (B_upl)


# In[59]:


plt.plot(X_std,Y_data,'ro')
plt.plot(np.sort(X_std),np.sort(X_r.dot(B_up)),linewidth=2.5,label=r'$Ridge \quad \beta=[0.42,0.50]$')
plt.plot(np.sort(X_std),np.sort(1/(1+np.exp(-X_r.dot(B_upl)))),linewidth=2.5, label=r'$Log \quad \beta=[3.2,0.08]$')
plt.legend(loc='lower right')
plt.suptitle('Logistic regression Vs Ridge regression', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True)


# In[60]:


plt.show()


# 5) Add an additional data point ($X_{15}$, $Y_{15}$ )$= $(3, 1) to the standardized set (X, Y ) and repeat the previous part.

# In[25]:

X_nd=np.append(X_std,[3])
X_twod=np.ones(15)
Y_nd=np.append(Y_data,[1])


# In[27]:

X_d=np.vstack((X_nd,X_twod)).T


# In[61]:

B_upd=J_update(X_d,Y_nd,B_in,0.07,3)
print (B_upd)
B_upld=l_update(X_d,Y_nd,B_in,0.07,3)
print (B_upld)


# In[64]:

plt.plot(X_nd,Y_nd,'ro')
plt.plot(np.sort(X_nd),np.sort(X_d.dot(B_upd)),linewidth=2,label=r'$Ridge \quad \beta=[0.33,0.46]$')
plt.plot(np.sort(X_nd),np.sort(1/(1+np.exp(-X_d.dot(B_upld)))),linewidth=2, label=r'$Log \quad \beta=[3.2,0.08]$')
plt.legend(loc='lower right')
plt.suptitle('Log Vs Ridge (Additional Data)', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True)


# In[65]:

plt.show()


# In[ ]:



