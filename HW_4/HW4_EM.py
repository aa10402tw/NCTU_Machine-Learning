from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import math
from math import e
from math import log
from numpy import linalg
import sys
import sympy
from sympy import *
from IPython.display import display, HTML
import random
from math import log
from math import exp

# %matplotlib inline
# %config IPCompleter.greedy=True
# sympy.init_printing(use_unicode=False, wrap_line=True)
np.set_printoptions(suppress=True)

from utils import *


# Load mnist data
print('Load Data...', end=' ')
imgs_train, labels_train = load_mnist(train=True)
imgs_test, labels_test = load_mnist(train=False)

# Change Image to Feature Vector
X_train, Y_train = img_to_vector(imgs_train), labels_train
X_test, Y_test = img_to_vector(imgs_test), labels_test

print('Finish!')

print(X_train.shape)
print(max(X_train[0]))

# E-Step : Compute the probabilities of cluster assignment (r_ik)
# M-step : Update parameters mu, pi givn r 

# X = [[x_11...x_1d], ..., [x_n1...x_nd]] where x_ij = 0/1 (success/fail)
# mu : [mu_1...mu_k] where mu_i is the vector of prob of success for cluster k , mu_i's shape = (1, D)
# pi : [pi_1 ... pi_k] where pi_i is the prob to draw cluster k
# r_iK : the prob(expectation) that Xi belong to cluster k
# Zi : [z_i1, ..., z_ik] binary k=dim data(assign to cluster k)

num_cluster = 10
K = num_cluster
X = np.copy(X_train)[:]
Y = np.copy(Y_train)[:]
N, D = X.shape
# init parameters mu & pi
mu = np.random.random((K, D))
pi = np.random.random((K, 1)) 
pi = pi / pi.sum()
r = np.zeros((N,K))
print(X.max())
print(X.min())

# 0~255 to 0 or 1
X[X<128.0] = 0
X[X>=128.0] = 1
print(X.max())
print(X.min())

from numpy import prod

def L2distance(A,B):
    A = A.reshape(prod(A.shape))
    B = B.reshape(prod(B.shape))
    dis = math.sqrt(np.dot(A-B, A-B))
    return dis

def EM(X, mu, pi, r, max_iter=100):
    
    N, D = X.shape
    K, _ = mu.shape
    
    new_mu = np.copy(mu)
    for it in range(max_iter):
        # E-Step : Compute the probabilities of cluster assignment (r_ik)
        for i in range(N):
            for k in range(K):
                r[i][k] = log(pi[k]) # Log scale
                for d in range(D):
                    xid = X[i][d]
                    try:
                        r[i][k] += log((mu[k][d]**xid) * ((1-mu[k][d])**(1-xid))+1e-7)
                    except:
                        print('domain error')
                        print(mu[k][d], xid)
                        print((mu[k][d]**xid) * ((1-mu[k][d])**(1-xid)))
            r[i] -= r[i].max()
            r[i] = np.exp(r[i]) # Exp, back to origin scale
            r[i] = r[i] / r[i].sum() # normalize to 1 
        Nk = r.sum(axis=0)  # prob to draw k-th cluster
        pi = Nk/Nk.sum()

        # M-step : Update parameters mu, pi givn r 
        for k in range(K):
            mu_k = 0
            for i in range(N):
                mu_k += r[i][k] * X[i]
            new_mu[k] = mu_k / Nk[k]
        diff = L2distance(new_mu, mu)
        print(diff) 
        mu = np.copy(new_mu)
        if diff < 1e-5:
            print('converge after %d iteration'%(it))
            break
    return mu, pi, r

def EM_inference(X, mu, pi):
    N, D = X.shape
    K, _ = mu.shape
    y_pred = np.zeros((N,))
    for i in range(N):
        for k in range(K):
            r[i][k] = log(pi[k]) # Log scale
            for d in range(D):
                xid = X[i][d]
                try:
                    r[i][k] += log((mu[k][d]**xid) * ((1-mu[k][d])**(1-xid))+1e-7)
                except:
                    print('domain error')
                    print(mu[k][d], xid)
                    print((mu[k][d]**xid) * ((1-mu[k][d])**(1-xid)))
#         print(r[i])
        y_pred[i] = np.argmax(r[i]) 
    return y_pred

mu, pi, r = EM(X, mu, pi, r)

y_pred = EM_inference(X, mu, pi)

from sklearn.metrics import confusion_matrix

count_y = [np.count_nonzero(Y == i) for i in range(10)]
count_y_pred = [np.count_nonzero(y_pred == i) for i in range(num_cluster)]
print(count_y)
print(count_y_pred)

print(confusion_matrix(y_pred, Y))

print(r.shape)
print(mu.shape)

print(mu.max())
print(mu.min())

for i in range(num_cluster):
    plt.subplot(2,5,i+1)
    p = mu[i].reshape((28,28))
    plt.imshow(p, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()