#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from numpy import prod
import pandas as pd
import numpy as np
import cv2


# In[2]:


import os
from glob import glob
img_dirs = ['att_faces//s'+ str(i) for i in range(1, 41)]
print(img_dirs)


# In[3]:


def read_dir_imgs(dir_path):
    imgs = []
    for img_path in glob( dir_path+'/*.pgm'):
        img = cv2.imread(img_path, 0)
        imgs.append(img)
    return imgs

def vec2img(vec, img_size=(112, 92)):
    return vec.reshape(img_size)

def img2vec(img):
    return img.reshape((prod(img.shape)))

imgs = []
for dir_path in img_dirs:
    imgs += read_dir_imgs(dir_path)

imgs = np.array(imgs)
print(imgs.shape)
imgs = imgs.reshape( (prod(imgs.shape[:1]), prod(imgs.shape[1:])) ).T
print(imgs.shape)
print(OUO)

# In[4]:


mean_vector = imgs.mean(1)

mean_face = vec2img(mean_vector)
plt.imshow(mean_face, cmap = 'gray'), plt.show()

diff_imgs = imgs - np.tile(np.array([mean_vector]).T, (1, 400))
T_trans_T = np.cov(diff_imgs.T)


# In[5]:


print(imgs.shape)

print(np.tile(np.array([mean_vector]).T, (1, 400)))


# In[ ]:





# In[6]:


print(imgs.shape)
print(T_trans_T.shape)


# In[7]:


T_trans_T.mean()


# # Compute EigenFace

# In[8]:


from math import exp
from numpy import linalg as LA

eigenValues, eigenVectors = np.linalg.eig(T_trans_T)

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]


k = 25
eigenValues = eigenValues[:k]
eigenVectors = eigenVectors[:, 0:k]

# get real eigen vector(eigen faces)
A = diff_imgs
eigenVectors = np.matmul(A, eigenVectors)
eigen_faces = np.copy(eigenVectors)

plt.figure(figsize=(6, 7.5))
n_row = int(k**(1/2))
n_col = int(k**(1/2))
for i in range(k):
    eigen_faces[:, i] =  eigenVectors[:,i] / np.linalg.norm(eigenVectors[:,i])
eigen_faces = eigen_faces.T
for i in range(k):
    face = vec2img(eigen_faces[i])
    plt.subplot(n_row ,n_col, i+1)
    plt.imshow(face, cmap = 'gray'), plt.xticks([]) , plt.yticks([])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.05)
plt.show()


# # Reconstruction 

# In[9]:


plt.figure(figsize=(20, 10))
id_list = [2,3,7,11,12,16, 27,36, 28, 29]

for img_cnt in range(10):
    img = cv2.imread('att_faces//s%d//4.pgm'%(id_list[img_cnt]), -1)
    plt.subplot(4, 10, 1+img_cnt)
    
    
    plt.imshow(img, cmap='gray'), plt.xticks([]) , plt.yticks([])
    
    img_vec = img2vec(img)
    diff_vec = img_vec - mean_vector
    diff_face = vec2img(diff_vec)
    #plt.imshow(diff_face, cmap='gray'), plt.xticks([]) , plt.yticks([])
    weights = []
    for i in range(k):
        weights.append(np.dot(diff_vec, eigen_faces[i]))
    reconstruct_vec = np.zeros(mean_vector.shape)
    
    for i in range(k):
        reconstruct_vec = reconstruct_vec + weights[i] * eigen_faces[i]
        reconstruct_face = vec2img(reconstruct_vec+mean_vector)
        
        if i+1 == 5:
            plt.subplot(4, 10, 11+img_cnt)
            plt.imshow(reconstruct_face, cmap='gray'), plt.xticks([]) , plt.yticks([])
            
        if i+1 == 15:
            plt.subplot(4, 10, 21+img_cnt)
            plt.imshow(reconstruct_face, cmap='gray'), plt.xticks([]) , plt.yticks([])
        if i+1 == k:
            plt.subplot(4, 10, 31+img_cnt)
            plt.imshow(reconstruct_face, cmap='gray'), plt.xticks([]) , plt.yticks([])
plt.show()


# plt.imshow(diff_face, cmap='gray'), plt.xticks([]) , plt.yticks([])
# plt.title('Diff')
# plt.show()

# # np.dot(diff_vec, eigen_faces)


# plt.figure(figsize=(10, 10))

# plt.show()

# reconstruct_face = vec2img(reconstruct_vec + mean_vector)
# plt.imshow(reconstruct_face, cmap='gray'), plt.xticks([]) , plt.yticks([])
# plt.title('Reconstruction(#eigFace=%d)'%(k))
# plt.show()


# In[ ]:





# In[ ]:




