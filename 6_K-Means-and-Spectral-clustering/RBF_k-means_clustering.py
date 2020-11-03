#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import exp
from tqdm import tqdm_notebook as tqdm

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


class RBF_kmeans_clustering:
    
    def __init__(self, k=1, gamma=1):
        self.k = k
        self.num_img = 0
        self.gamma = gamma
        
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
        # centroids = self.calculate_centroids(X, X_membership)
        COLORS = ['blue', 'red', 'green', 'black']
        COLORS2 = ['lightblue', 'orange', 'lightgreen', 'gray']
        for c, X_ in enumerate(X_class):
            plt.scatter(X_[:, 0], X_[:, 1], marker='.', c=COLORS2[c], s=10)
            # plt.scatter(centroids[c][0], centroids[c][1], s=50, c=COLORS[c], marker='x', label='centroid')
        plt.title(title)
        if 'Init' in title:
            for i in range(4):
                plt.savefig("imgs/file%02d.png" % self.num_img)
                self.num_img += 1 
        plt.savefig("imgs/file%02d.png" % self.num_img)
        # plt.legend(labels=['data', 'centroid'],  loc='upper right')
        plt.show()
        
        self.num_img += 1
    
    def compute_RBF_kernel(self, X, gamma=5):
        # k(x1,x2) = exp(-gamma*length(x1-x2)**2)
        RBF_kernel = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                RBF_kernel[i][j] = exp(-gamma* np.dot((X[i]-X[j]), (X[i]-X[j])))
        return rbf_kernel(X, gamma=gamma)
        
    def getMembershipMatrix(self, X_membership):
        K = self.k
        N = X_membership.shape[0]
        a = np.zeros((K, N), dtype=np.int32)
        for k in range(K):
            for n in range(N):
                a[k][n] = 1 if X_membership[n] == k else 0
        return a

    def fit(self, X, method=None, visulaize=True, max_iter=100):
        
        # init cluster
        X_membership = self.initMember(X, method)
        old_membership = np.copy(X_membership)
        
        membershipMatrix = self.getMembershipMatrix(X_membership)
        
        # Compute kernel
        RBF_kernrl = self.compute_RBF_kernel(X, self.gamma)
        
        self.draw(X, X_membership, title='Init (%s) k=%d, gamma=%.2f'%(method, self.k, self.gamma))
        for it in range(max_iter):
            
            # assign new membership based on kernel (skip for compute centroid)
            self.precompute_third_term(X, membershipMatrix, RBF_kernrl)
            X_membership = self.assign_membership(X, membershipMatrix, RBF_kernrl)
            membershipMatrix = self.getMembershipMatrix(X_membership)
            
            if(visulaize):
                self.draw(X, X_membership, title='Interation %d'%(it+1))
            
            # Determine convergance
            if np.count_nonzero(old_membership == X_membership) >= X.shape[0]*0.99:
                break
            else:
                old_membership = np.copy(X_membership)
        self.draw(X, X_membership, title='Converged (%d iter, gamma=%.2f)'%(it+1, self.gamma))
        return X_membership
        
    def assign_membership(self, X, membershipMatrix, RBF_kernrl):
        X_membership = np.zeros((X.shape[0]), dtype=np.int32)
        
        for ix, x in enumerate(X):
            min_dist = self.distance_to_cluster(X, membershipMatrix, RBF_kernrl, ix, 0)
            for cluster in range(self.k):
                cur_dist = self.distance_to_cluster(X, membershipMatrix, RBF_kernrl, ix, cluster)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    X_membership[ix] = cluster
                    
        return X_membership
    
    def precompute_third_term(self, X, a, RBF_kernrl):
        # Compute third term
        self.third_term = []
        for k in range(self.k):
            c = np.count_nonzero(a[k])
            third_term = 0
            for p in range(X.shape[0]):
                for q in range(X.shape[0]):
                    third_term += a[k][p]*a[k][q]*RBF_kernrl[p][q]
            third_term *= (1/c)**2
            self.third_term.append(third_term)
    
    
    def distance_to_cluster(self, X, membershipMatrix, RBF_kernrl, x_j, cluster_k):
        a = membershipMatrix
        j = x_j
        k = cluster_k
        c = np.count_nonzero(a[k])
        
        # Compute second term
        second_term = 0
        for n in range(X.shape[0]):
            second_term += a[k][n] * RBF_kernrl[j][n]
        second_term *= 2/c
        third_term = self.third_term[k]
        return RBF_kernrl[j][j] - second_term + third_term
    
    def distance(self, x1, x2):
        return np.sqrt(np.dot((x1-x2), (x1-x2)))


# In[10]:


init_methods = ['random', 'Dist2Origin', 'Dist2Center']
k_range = [2,3,4]
gamma = 0.5
for k in k_range:
    for method in init_methods:
        model = RBF_kmeans_clustering(k=k, gamma=gamma)
        membership = model.fit(X_moon[:], method=method, visulaize=False)


# In[7]:


# 'random', 'Dist2Origin', 'Dist2Center'
gamma = 0.5
model = RBF_kmeans_clustering(k=2, gamma=gamma)
membership = model.fit(X_circle[:], method='Dist2Center', visulaize=False)


# In[ ]:


gamma = 0.5
model = RBF_kmeans_clustering(k=4, gamma=gamma)
membership = model.fit(X_circle[:], method='Dist2Origin', visulaize=True)


# In[8]:


gamma = 0.5
model = RBF_kmeans_clustering(k=4, gamma=gamma)
membership = model.fit(X_circle[:], method='Dist2Center', visulaize=True)


# In[ ]:




