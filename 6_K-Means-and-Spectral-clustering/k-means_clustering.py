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

X_moon = np.array(data_moon)
X_circle = np.array(data_circle)


# In[3]:


class kmeans_clustering:
    
    def __init__(self, k=1):
        self.k = k
        self.num_img = 0
        
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
        
    def draw(self, X, X_membership, title=''):
        X_class = [X[X_membership==c] for c in range(self.k)]
        centroids = self.calculate_centroids(X, X_membership)
        COLORS = ['blue', 'red', 'green', 'black']
        COLORS2 = ['lightblue', 'orange', 'lightgreen', 'gray']
        for c, X_ in enumerate(X_class):
            plt.scatter(X_[:, 0], X_[:, 1], marker='.', c=COLORS2[c], s=10)
            plt.scatter(centroids[c][0], centroids[c][1], s=50, c=COLORS[c], marker='x', label='centroid')
        plt.title(title)
        if 'Init' in title:
            for i in range(4):
                plt.savefig("imgs/file%02d.png" % self.num_img)
                self.num_img += 1 
        plt.savefig("imgs/file%02d.png" % self.num_img)
        plt.legend(labels=['data', 'centroid'],  loc='upper right')
        plt.show()
        
        self.num_img += 1

    def fit(self, X, method=None, visulaize=True, max_iter=100):
        # random init member 
        X_membership = self.initMember(X, method)
        old_membership = np.copy(X_membership)
        self.draw(X, X_membership, title='Init (%s) k=%d'%(method, self.k))
        for it in range(max_iter):
            # calculate centroids based on membership
            self.centroids = self.calculate_centroids(X, X_membership)
            
            # assign membership based on centroids
            X_membership = self.assign_membership(X, self.centroids)

            if(visulaize):
                self.draw(X, X_membership, title='Interation %d'%(it+1))
            if np.count_nonzero(old_membership == X_membership) >= X.shape[0]*0.9999:
                break
            else:
                old_membership = np.copy(X_membership)
        
        self.draw(X, X_membership, title='Converged (%d iter)'%(it+1))
        return X_membership
    
    def calculate_centroids(self, X, X_membership):
        centroids = np.zeros( (self.k, *(X.shape[1:])) )
        num_data = [0 for x in range(self.k)]
    
        for x, membership in zip(X, X_membership):
            # updata by weighted sum
            centroids[membership] = (num_data[membership]*centroids[membership] + x) / (num_data[membership]+1)
            num_data[membership] += 1
        return centroids
        
    def assign_membership(self, X, centroids):
        X_membership = np.zeros((X.shape[0]), dtype=np.int32)
        for ix, x in enumerate(X):
            min_dist = self.distance(x, centroids[0])
            for ic, centroid in enumerate(centroids):
                if self.distance(x, centroid) < min_dist:
                    min_dist = self.distance(x, centroid)
                    X_membership[ix] = ic
        return X_membership
    

    
    def distance(self, x1, x2):
        return np.sqrt(np.dot((x1-x2), (x1-x2)))
    
    def predict(self, X):
        X_membership = self.assign_membership(X, self.centroids)
        return X_membership


# In[16]:


# method choice = 'random', 'Dist2Origin', 'Dist2Center'
model = kmeans_clustering(k=2)
membership = model.fit(X_moon, method='Dist2Center', visulaize=False)


# In[17]:


model = kmeans_clustering(k=4)
model.fit(X_moon, method='Dist2Origin')


# In[20]:


model = kmeans_clustering(k=4)
model.fit(X_moon, method='Dist2Center')


# In[15]:


# method choice = 'random', 'Dist2Origin', 'Dist2Center'
model = kmeans_clustering(k=2)
model.fit(X_circle, method='Dist2Center', visulaize=False)


# In[28]:


model = kmeans_clustering(k=4)
model.fit(X_circle, method='Dist2Origin')


# In[31]:


model = kmeans_clustering(k=4)
model.fit(X_circle, method='Dist2Center')


# In[31]:


print(X_moon.shape)


# In[ ]:




