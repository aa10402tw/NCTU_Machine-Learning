import math
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

METHODS = ['12_uniform', 'Box-Muller_Tranform', 'uniform_inverse']

def CLT_std_norm_rand():
    random_nums = [random.uniform(0, 1) for x in range(12)] 
    std_norm_rand = sum(random_nums) - 6
    return std_norm_rand

def gaussian_data_generator(mean, var, method=0):
    if method==0:
        std_norm_rand = CLT_std_norm_rand()
    norm_rand = mean + (std_norm_rand * var**(1/2))
    return norm_rand

def polynomial_linear_data_generator(num_basis, a, w, x):
    # x = [random.uniform(-10, 10) * 20 for x in range(num_basis)]
    # x[0] = 1
    e = gaussian_data_generator(0, a)
    y = np.dot(x, w) + e
    return y

def test_g():
    mean = 4
    var = 1
    num_bins = 100
    rand_nums = []
    for i in range(10000):
        std_norm_rand = gaussian_data_generator(mean, var)
        rand_nums.append(float(int(std_norm_rand*num_bins*10))/(num_bins*10))

    import matplotlib.pyplot as plt
    from scipy.stats import norm
    range_min = mean - 3 * var**(1/2)
    range_max = mean + 3 * var**(1/2)
    # plt.subplot(121)
    plt.hist(rand_nums, bins=int(num_bins), normed=1, alpha=0.5, range=(range_min, range_max))

    # plt.subplot(122)
    x_axis = np.arange(range_min, range_max, 10/num_bins)
    plt.plot(x_axis, norm.pdf(x_axis, mean, var**(1/2)))
    title = "mean={}, var={}".format(mean, var)
    plt.title(title)

    plt.show()

def test_p():
    w = [1, 0, 0, 0.1]
    noisy = 1
    num_basis = len(w)
    num_points = 100
    xs = np.arange(-10, 10, 20/num_points)
    phi_xs = [[x**(d) for d in range(num_basis)] for x in xs]
    # Poly
    ys = []
    for i in range(num_points):
        x = phi_xs[i]
        y = polynomial_linear_data_generator(num_basis, 0, w, x)
        ys.append(y)
    plt.plot(xs, ys, label='Without Noisy')

    # Noisy 
    ys = []
    for i in range(num_points):
        x = phi_xs[i]
        y = polynomial_linear_data_generator(num_basis, noisy, w, x)
        ys.append(y)
    plt.plot(xs, ys, label='With Noisy')

    title = "y = "
    for d, dw in enumerate(w):
        title += "{}(x^{})".format(dw, d)
        if d < len(w)-1:
            title += " + "
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    test_g()
    test_p()


# import numpy as np
# import matplotlib.pyplot as plt

# #generate from uniform dist
# import math
# import random
# import sys
# import traceback


# class RndnumBoxMuller:
#     M     = 100        # Average
#     S     = 10       # Standard deviation
#     N     = 10000     # Number to generate
#     SCALE = N // 100  # Scale for histgram

#     def __init__(self):
#         self.hist = [0 for _ in range(self.M * 5)]

#     def generate_rndnum(self):
#         """ Generation of random numbers """
#         try:
#             for _ in range(self.N):
#                 res = self.__rnd()
#                 self.hist[res[0]] += 1
#                 self.hist[res[1]] += 1
#         except Exception as e:
#             raise

#     def display(self):
#         """ Display """
#         try:
#             for i in range(0, self.M * 2 + 1):
#                 print("{:>3}:{:>4} | ".format(i, self.hist[i]), end="")
#                 for j in range(1, self.hist[i] // self.SCALE + 1):
#                     print("*", end="")
#                 print()
#         except Exception as e:
#             raise

#     def __rnd(self):
#         """ Generation of random integers """
#         try:
#             r_1 = random.random()
#             r_2 = random.random()
#             x = self.S \
#               * math.sqrt(-2 * math.log(r_1)) \
#               * math.cos(2 * math.pi * r_2) \
#               + self.M
#             y = self.S \
#               * math.sqrt(-2 * math.log(r_1)) \
#               * math.sin(2 * math.pi * r_2) \
#               + self.M
#             return [math.floor(x), math.floor(y)]
#         except Exception as e:
#             raise


# if __name__ == '__main__':
#     try:
#         obj = RndnumBoxMuller()
#         obj.generate_rndnum()
#         obj.display()
#     except Exception as e:
#         traceback.print_exc()
#         sys.exit(1)

