#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import exp

def read_data(file_name):
    datas = []
    with open(file_name) as f:
        for line in f:
            data = line.split(',')
            data[0] = float(data[0])
            data[1] = float(data[1])
            datas.append(data)
    return datas


# In[2]:


data_moon = read_data('moon.txt')
data_circle = read_data('circle.txt')

X = np.array(data_moon+data_circle)
X_moon = np.array(data_moon)
X_circle = np.array(data_circle)


# In[3]:


class k_means_clustering:
    
    
    def __init__(self, k=1):
        self.k = k
        
    def initMember(self, X, method='random'):
        N = X.shape[0]
        X_membership = np.zeros((N), dtype=np.int32)

        if method == 'random':
            for i in range(N):
                X_membership[i] = np.random.randint(self.k)
                
        elif method == 'Dist2Origin':
            origin = np.zeros(X[0].shape)
            X_to_origin =  [self.distance(x, origin) for x in X]
            X_to_origin_order = np.argsort(X_to_origin)
            num_data_per_cluster = N//self.k
            for cluster in range(self.k):
                for idx in X_to_origin_order[(cluster)*num_data_per_cluster:(cluster+1)*num_data_per_cluster]:
                    X_membership[idx] = cluster
        
        elif method == 'Dist2Center':
            center = np.mean(X, axis=0)
            X_to_origin =  [self.distance(x, center) for x in X]
            X_to_origin_order = np.argsort(X_to_origin)
            num_data_per_cluster = N//self.k
            for cluster in range(self.k):
                for idx in X_to_origin_order[(cluster)*num_data_per_cluster:(cluster+1)*num_data_per_cluster]:
                    X_membership[idx] = cluster
        return X_membership
        
    def fit(self, X):
        
        # random init membership
        X_membership = self.initMember(X, method='random')
        
        while True:
            old_membership = np.copy(X_membership)
            
            # calculate centroids based on membership
            self.centroids = self.calculate_centroids(X, X_membership)
            
            # assign membership based on centroids
            X_membership = self.assign_membership(X, self.centroids)

            if np.array_equal(old_membership, X_membership):
                break
        return X_membership
        
    def assign_membership(self, X, centroids):
        X_membership = np.zeros((X.shape[0]), dtype=np.int32)
        for ix, x in enumerate(X):
            min_dist = self.distance(x, centroids[0])
            for ic, centroid in enumerate(centroids):
                if self.distance(x, centroid) < min_dist:
                    min_dist = self.distance(x, centroid)
                    X_membership[ix] = ic
        return X_membership
    
    def calculate_centroids(self, X, X_membership):
        centroids = np.zeros( (self.k, *(X.shape[1:])) )
        num_data = [0 for x in range(self.k)]
    
        for x, membership in zip(X, X_membership):
            # updata by weighted sum
            centroids[membership] = (num_data[membership]*centroids[membership] + x) / (num_data[membership]+1)
            num_data[membership] += 1

        return centroids
    
    def distance(self, x1, x2):
        return np.sqrt(np.dot((x1-x2), (x1-x2)))
    
    
    def predict(self, X):
        X_membership = self.assign_membership(X, self.centroids)
        return X_membership


# In[4]:


# L= D−W 
# find first k eigenvector 
# Do k-means on that

import sklearn
from sklearn import metrics
rbf_kernel = metrics.pairwise.rbf_kernel

def compute_RBF_kernel(X, gamma=5):
    # k(x1,x2) = exp(-gamma*length(x1-x2)**2)
    RBF_kernel = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            RBF_kernel[i][j] = exp(-gamma* np.dot((X[i]-X[j]), (X[i]-X[j])))
    return rbf_kernel(X, gamma=gamma)

from numpy import linalg as LA
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class Spectral_clustering:
    
    def __init__(self, n_cluster=2, gamma=1):
        self.n_cluster = n_cluster
        self.gamma = gamma

    def fit(self, X, gamma=1):
        # 0. Define similarity matrix W and D
        W = compute_RBF_kernel(X, gamma=self.gamma)
        D = np.zeros((X.shape[0], X.shape[0]))
        for d in range(X.shape[0]):
            D[d][d] = W[d].sum()
        
        # 1. Graph Laplacian L = D - W
        L = D - W
        
        # Get first k eigenvector 
        eigenValues, eigenVectors = LA.eig(L)
        idx = eigenValues.argsort() 
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        U = eigenVectors[:, 1:self.n_cluster+1]
        
        # Do k-means 
        kmeans = k_means_clustering(k=2)
        membership = kmeans.fit(U)
        print(membership)
        self.draw(X, membership)
        
            
    def draw(self, X, X_membership):
        X_class = [X[X_membership==c] for c in range(self.n_cluster)]
        COLORS = ['blue', 'red', 'green', 'black', 'gray']
        COLORS2 = ['lightblue', 'orange', 'lightgreen', 'black']
        for c, X_ in enumerate(X_class):
            plt.scatter(X_[:, 0], X_[:, 1], marker='.', c=COLORS2[c], s=50)
        
        plt.title('Gamma = %f'%self.gamma)
        plt.show()


# In[7]:


gamma_range = [(i) for i in range(10, 100, 10)]
for gamma in gamma_range:
    model = Spectral_clustering(n_cluster=2, gamma=gamma)
    membership = model.fit(X_circle[:])


# In[6]:


for gamma in range(15, 25):
    model = Spectral_clustering(n_cluster=2, gamma=gamma)
    membership = model.fit(X_moon[:])


# In[103]:


import sklearn
kernel = rbf_kernel(X_moon, gamma=1)
membership = sklearn.cluster.spectral_clustering(n_clusters=2, affinity=kernel, n_init=100)


# In[104]:


def draw(X, X_membership):
    X_class = [X[X_membership==c] for c in range(2)]
    COLORS = ['blue', 'red', 'green', 'black', 'gray']
    COLORS2 = ['lightblue', 'orange', 'lightgreen', 'black']
    for c, X_ in enumerate(X_class):
        plt.scatter(X_[:, 0], X_[:, 1], marker='.', c=COLORS2[c], s=50)
    plt.show()
draw(X_moon, membership)


# In[ ]:




