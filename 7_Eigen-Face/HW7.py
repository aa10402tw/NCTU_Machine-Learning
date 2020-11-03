import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)

df_X = pd.read_csv('X_train.csv', header=None)
df_Y = pd.read_csv('T_train.csv', header=None)

X_train = df_X.values
Y_train = df_Y.values
Y_train = Y_train.reshape(Y_train.shape[0])


print(X_train.shape)
print(Y_train.shape)

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    def transform(self, X):
        X_high = np.copy(X)
        mean_mat = np.tile(self.mean_vec, (X.shape[0],1))
        diff_mat = X_high - mean_mat
        # Project from high to low
        X_low = np.matmul(diff_mat, self.W)
        return np.real(X_low)
    
    def fit(self, X):
        X_high = np.copy(X)
        mean_vec = np.mean(X_high, 0)
        mean_mat = np.tile(mean_vec, (X.shape[0],1))
        diff_mat = X_high - mean_mat
        cov_mat = np.cov(diff_mat.T)
        self.mean_vec = mean_vec
        
        # Compute eigenpairs of cov mat
        eigenValues, eigenVectors = np.linalg.eig(cov_mat)
        idx = eigenValues.argsort()[::-1]   
        W = eigenVectors[:,idx][:, :self.n_components]
        W = W * -1 
        self.W = W
        return self
pca = PCA(n_components=2)
X_low_pca = pca.fit(X_train).transform(X_train)


## Mine
class LDA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = 0
        self.std = 1
    
    def transform(self, X):
        X_high = np.copy(X)
        X_high = (X_high - self.mean) / self.std 
        # Project from high to low
        X_low = np.matmul(X_high, self.W)
        return np.real(X_low)
        
    def fit(self, X, Y):
        N, dim = X.shape
        X_high = np.copy(X)
        self.mean = X_high.mean()
        self.std = X_high.std()
        X_high = (X_high - self.mean) / self.std 
        
        # Compute mean for each class (mj, nj)
        mean_vectors = []
        for c in set(Y):
            mean_vectors.append( np.mean(X_high[Y==c], axis=0) )
        self.mean_vectors = mean_vectors
        
        # Compute within-class scatter
        SW = np.zeros( (dim,dim) )
        for c, mv in zip(set(Y), mean_vectors):
            within_class_scattter = np.zeros((dim, dim))
            for xi in X_high[Y==c]:
                xi = xi.reshape(-1, 1) # make vec to mat
                mj = mv.reshape(-1, 1) # make vec to mat
                within_class_scattter += np.matmul(xi-mj, (xi-mj).T)
            SW += within_class_scattter
    
        # Compute between-class scatter
        SB = np.zeros( (dim,dim) )
        m = np.mean(X_high, axis=0).reshape(-1, 1)
        for c, mv in zip(set(Y), mean_vectors):
            nj = X_high[Y==c].shape[0]
            mj = mv.reshape(-1, 1) # make vec to mat
            SB += nj * np.matmul((mj-m), (mj-m).T)
            
        # Compute W using first k eigenvetor of inv(SW)*SB
        mat = np.dot(np.linalg.pinv(SW), SB)
        eigenValues, eigenVectors = np.linalg.eig(mat)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        W = np.real(eigenVectors[:, 0:self.n_components])
        W /= np.linalg.norm(W, axis=0)
        self.W = W
        return self
lda = LDA(n_components=2)
X_low_lda = lda.fit(X_train, Y_train).transform(X_train)


# In[55]:


from math import exp
from numpy import linalg as LA

def compute_linear_kernel(X):
    # k(x1, x2) = x1^T * x2
    N, dim = X.shape
    kernel = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            kernel[i][j] = np.dot(X[i], X[j])
    return kernel
            
def compute_RBF_kernel(X, gamma=5):
    # k(x1,x2) = exp(-gamma*length(x1-x2)**2)
    N, dim = X.shape
    RBF_kernel = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            RBF_kernel[i][j] = exp(-gamma* np.dot((X[i]-X[j]), (X[i]-X[j])))
    return RBF_kernel

def compute_linear_rbf_kernel(X, gamma=5):
    rbf_kernel = compute_RBF_kernel(X, gamma)
    linear_kernel = compute_linear_kernel(X)
    return linear_kernel + rbf_kernel 
    
# linear_kernel = compute_linear_kernel(X_low_pca)
# rbf_kernel =  compute_RBF_kernel(X_low_pca)
# linear_rbf_kernel = compute_linear_rbf_kernel(X_low_pca)


# In[9]:


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

class Spectral_clustering_rationCut:
    
    def __init__(self, n_cluster=2, gamma=1):
        self.n_cluster = n_cluster
        self.gamma = gamma
        
    def fit(self, X, kernel='rbf'):
        # 0. Define similarity matrix W and D
        W = []
       
        if kernel == 'linear':
            W = compute_linear_kernel(X)
        elif kernel == 'rbf':
            W = compute_RBF_kernel(X, self.gamma)
        elif kernel == 'rbf_linear':
            W = kernel = compute_linear_rbf_kernel(X, self.gamma)

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
        kmeans = k_means_clustering(k=self.n_cluster)
        membership = kmeans.fit(U)
        return membership
    
from scipy.linalg import sqrtm   

class Spectral_clustering_normCut:
    
    def __init__(self, n_cluster=2, gamma=1):
        self.n_cluster = n_cluster
        self.gamma = gamma
        
    def fit(self, X, kernel='rbf'):
        # 0. Define similarity matrix W and D
        W = []
       
        if kernel == 'linear':
            W = compute_linear_kernel(X)
        elif kernel == 'rbf':
            W = compute_RBF_kernel(X, self.gamma)
        elif kernel== 'rbf_linear':
            W = kernel = compute_linear_rbf_kernel(X, self.gamma)
        D = np.zeros((X.shape[0], X.shape[0]))
        for d in range(X.shape[0]):
            D[d][d] = W[d].sum()
        
        # 1. Graph Laplacian L = D - W, L_norm = D^(-1/2) L D^(-1/2)
        L = D - W
        D_inv_sqrt = np.linalg.pinv(sqrtm(D)) 
        L = np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt) 
        
        # Get first k eigenvector 
        eigenValues, eigenVectors = LA.eig(L)
        idx = eigenValues.argsort() 
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        U = eigenVectors[:, 1:self.n_cluster+1]
        
        # Do k-means 
        kmeans = k_means_clustering(k=self.n_cluster)
        membership = kmeans.fit(U)
        return membership



import time
start = time.time()
model = Spectral_clustering_rationCut(n_cluster=5, gamma=1)
Y_pca_ratioCut = model.fit(X_low_pca, kernel='rbf')
print('RatioCut Done')

print(time.time() - start)
start = time.time()
model = Spectral_clustering_normCut(n_cluster=5, gamma=1)
Y_pca_normCut= model.fit(X_low_pca, kernel='rbf')
print('NormCut Done')
print(time.time() - start)

start = time.time()
model = Spectral_clustering_rationCut(n_cluster=5, gamma=1)
Y_lda_ratioCut = model.fit(X_low_lda, kernel='rbf')
print('RatioCut Done')

print(time.time() - start)
start = time.time()
model = Spectral_clustering_normCut(n_cluster=5, gamma=1)
Y_lda_normCut= model.fit(X_low_lda, kernel='rbf')
print('NormCut Done')
print(time.time() - start)


def plot_cluster(X, Y, title=''):
    for c in (set(Y)):
        X_c = X[Y==c]
        plt.scatter(X_c[:, 0], X_c[:, 1], label=str(c))
    plt.title(title)

plt.figure(figsize=(16, 4))
plt.subplot(131), plot_cluster(X_low_pca, Y_train, title='PCA Ground Truth'), plt.legend(loc='best')
plt.subplot(132), plot_cluster(X_low_pca, Y_pca_ratioCut, title='PCA Ratio Cut'), plt.legend(loc='best')
plt.subplot(133), plot_cluster(X_low_pca, Y_pca_normCut, title='PCA Normalized Cut'), plt.legend(loc='best')
plt.show()

plt.figure(figsize=(16, 4))
plt.subplot(131), plot_cluster(X_low_lda, Y_train, title='LDA Ground Truth'), plt.legend(loc='best')
plt.subplot(132), plot_cluster(X_low_pca, Y_lda_ratioCut, title='PCA Ratio Cut'), plt.legend(loc='best')
plt.subplot(133), plot_cluster(X_low_pca, Y_lda_normCut, title='PCA Normalized Cut'), plt.legend(loc='best')

plt.show()


    
def plot_sv(X, Y, sv_index, title=''):
    
    non_sv_X_c = [[] for i in set(Y)] 
    sv_X_c = [[] for i in set(Y)] 
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for i, x in enumerate(X):
        c = Y[i] - 1
        if i in sv_index:
            sv_X_c[c].append(x) 
        else:
            non_sv_X_c[c].append(x)
            
    for i in range(len(set(Y))):
        non_sv_X_c[i] = np.array(non_sv_X_c[i])
        sv_X_c[i] = np.array(sv_X_c[i])
    
    for c in range(len(set(Y))):
        plt.scatter(non_sv_X_c[c][:, 0], non_sv_X_c[c][:, 1], label=str(c+1), marker='.', c=colors[c], s=1)
        plt.scatter(sv_X_c[c][:, 0], sv_X_c[c][:, 1], marker='^', c=colors[c], s=30)
    plt.title(title)
    
from sklearn import svm

svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_low_pca, Y_train) 
y_pred_svm_linear = svm_linear.predict(X_low_pca)
sv_linear_index = svm_linear.support_
print(svm_linear.n_support_)

svm_poly = svm.SVC(kernel='poly')
svm_poly.fit(X_low_pca, Y_train) 
y_pred_poly_rbf = svm_poly.predict(X_low_pca)
sv_poly_index = svm_poly.support_
print(svm_poly.n_support_)


svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_low_pca, Y_train) 
y_pred_svm_rbf = svm_rbf.predict(X_low_pca)
sv_rbf_index = svm_rbf.support_
print(svm_rbf.n_support_)

svm_linear_rbf = svm.SVC(kernel='precomputed')
kernel = compute_linear_rbf_kernel(X_train, gamma=1/784)
svm_linear_rbf.fit(kernel, Y_train) 
# y_pred_svm_rbf = svm_linear_rbf.predict(X_low_pca)
sv_linear_rbf_index = svm_linear_rbf.support_
print(svm_linear_rbf.n_support_)


plt.figure(figsize=(16, 12))
plt.subplot(221), plot_sv(X_low_pca, Y_train, sv_linear_index, title='Linear SVM'), plt.legend(loc='best')
plt.subplot(222), plot_sv(X_low_pca, Y_train, sv_poly_index, title='Poly SVM'), plt.legend(loc='best')
plt.subplot(223), plot_sv(X_low_pca, Y_train, sv_rbf_index, title='RBF SVM'), plt.legend(loc='best')
plt.subplot(224), plot_sv(X_low_pca, Y_train, sv_linear_rbf_index, title='Linear RBF SVM'), plt.legend(loc='best')





