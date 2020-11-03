#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pylab

from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[2]:


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


# In[3]:


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    pbar = tqdm(total=n, ncols=800)
    pbar.set_description('Computing P-values')
    for i in range(n):

        # Print progress
        pbar.update()
#         if i % 500 == 0:
#             print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    pbar.close()
    return P


# In[4]:


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


# In[5]:


def tSNE(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    
    P_copy = np.copy(P)
    
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)
    
    

    # Run iterations
    pbar = tqdm(total = max_iter)
    for iter in range(max_iter):

        # Compute pairwise affinities (### MODIFY !!)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        Q_copy = np.copy(Q)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) +                 (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        C = np.sum(P * np.log(P / Q))
        pbar.set_postfix(Error=C)
        pbar.update()
#         if (iter + 1) % 10 == 0:
#             C = np.sum(P * np.log(P / Q))
#             print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    pbar.close()
    return Y, P_copy, Q_copy


# In[13]:


def symSNE(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    
    P_copy = np.copy(P)
    
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    
    
    # Run iterations
    pbar = tqdm(total = max_iter)
    for iter in range(max_iter):

        # Compute pairwise affinities (### MODIFY !!)
        sum_Y = np.sum(np.square(Y), axis=1) # yi^T * yi
        num = -2. * np.dot(Y, Y.T)  # -2 *  yj^T * yj
        num = np.exp(-1*np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0. # dia set to zeros
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        Q_copy = np.copy(Q)
        
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) +                 (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        C = np.sum(P * np.log(P / Q))
        pbar.set_postfix(Error=C)
        pbar.update()
#         if (iter + 1) % 10 == 0:
#             C = np.sum(P * np.log(P / Q))
#             print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 2.

    # Return solution
    pbar.close()
    return Y, P_copy, Q_copy


X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
perplexity_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Y, P, Q = tSNE(X, 2, initial_dims=50, perplexity=30.0)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()


high_distribution = P.reshape((2500*2500))
low_distribution = Q.reshape((2500*2500))

low_range = (low_distribution.min(), low_distribution.max())

plt.hist(high_distribution[:], bins=1000, density=True, log=True)
plt.title("High dimension similarity distribution(log scale)")
plt.show()

plt.hist(low_distribution[:], bins=1000, density=True, log=True, range=low_range)
plt.title("Low dimension similarity distribution(log scale)")
plt.show()



X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
perplexity_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Y, P, Q = symSNE(X, 2, initial_dims=50, perplexity=30.0)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()


high_distribution = P.reshape((2500*2500))
low_distribution = Q.reshape((2500*2500))


plt.hist(high_distribution[:], bins=1000, density=True, log=True)
plt.title("High dimension similarity distribution(log scale)")
plt.show()

plt.hist(low_distribution[:], bins=100, density=True, log=True, range=low_range)
plt.title("Low dimension similarity distribution(log scale)")
plt.show()



X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
perplexity_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for perplexity in perplexity_range:
    print("perplexity=", perplexity)
    Y, P, Q = tSNE(X, 2, 50, perplexity=perplexity)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()





