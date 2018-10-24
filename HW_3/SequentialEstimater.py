import math
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

from RandomDataGenerator import *



class Estimater:

    def __init__(self):
        pass

    def update(self):
        pass

    def estimate(self):
        pass

def test_estimater(mean, var, num_datas):
    model = Estimater()
    last_mean, last_var = 0, 0 
    for i in range(num_datas):
        data = gaussian_data_generator(mean, var)
        model.update(data)
        mean_estimated, var_estimated =  model.estimate()


if __name__ == '__main__':




