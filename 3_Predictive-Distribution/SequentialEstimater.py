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


class Estimator:

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


def test_estimator(mean, var, max_iters=100000, batch_size=1000, threshold='auto'):

    if threshold == 'auto':
        model = Estimator()
        for i in range(1, batch_size+1):
            data = gaussian_data_generator(mean, var)
            model.update(data)
        mean_estimated, var_estimated = model.estimate()
        threshold = 0.0005 * var_estimated

    model = Estimator()
    last_mean, last_var = 0, 0

    for i in range(1, max_iters+1):
        data = gaussian_data_generator(mean, var)
        model.update(data)
        mean_estimated, var_estimated = model.estimate()
        print('Data:%.4f' % data, '\tMean:%.4f' % mean_estimated, '\tVar:%.4f' % var_estimated)
        if i % batch_size == 0:
            diff_mean = abs(mean_estimated-last_mean)
            diff_var = abs(var_estimated-last_var)

            if diff_mean < threshold and diff_var < threshold:
                break
            last_mean, last_var = mean_estimated, var_estimated


if __name__ == '__main__':
    test_estimator(mean=0, var=1)
