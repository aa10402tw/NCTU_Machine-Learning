import math
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

from RandomDataGenerator import *


def mean(data):
    return sum(data) / len(data)


def var(data):
    m = mean(data)
    sum = 0
    for d in data:
        sum += (d - m) ** 2
    return sum / len(data)

# Use Welford's Online algorithm


class Estimater:

    def __init__(self):
        self.num_data = 0
        self.mean = 0
        self.var = 0

    def update(self, x):
        self.num_data += 1
        if self.num_data == 1:
            self.mean = x
            self.var = 0.0
        else:
            self.mean = self.mean + (x - self.mean) / self.num_data
            self.var = self.var + (x - self.mean)**2 / self.num_data - self.var / (self.num_data - 1)

    def estimate(self):
        return self.mean, self.var


def test_estimater(mean, var, max_iters):
    model = Estimater()
    last_mean, last_var = 0, 0
    for i in range(max_iters):
        data = gaussian_data_generator(mean, var)
        model.update(data)
        mean_estimated, var_estimated = model.estimate()
        # print(last_mean, last_var)
        print('Data:%.4f' % data, 'Mean:%.4f' % mean_estimated, '\tVar%.4f' % var_estimated)
        last_mean, last_var = mean_estimated, var_estimated

if __name__ == '__main__':
    test_estimater(mean=0, var=10, max_iters=1000)
