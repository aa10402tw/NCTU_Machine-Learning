import math
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

from RandomDataGenerator import *
# from SequentialEstimator import *


# def polynomial_linear_data_generator(num_basis, a, w, x):
#     # x = [random.uniform(-10, 10) * 20 for x in range(num_basis)]
#     # x[0] = 1
#     e = gaussian_data_generator(0, a)
#     y = np.dot(x, w) + e
#     return y


# if __name__ == '__main__':

#     w = [1, 0, 0, 0.1]
#     a = 1
#     num_basis = len(w)
#     num_points = 100
#     for i in range(num_points):
#         x = random.uniform(-10, 10)
#         x = [x**d for d in range(num_basis)]
#         y = polynomial_linear_data_generator(num_basis, a, w, x)

    



#     test_estimator(mean=0, var=1)
