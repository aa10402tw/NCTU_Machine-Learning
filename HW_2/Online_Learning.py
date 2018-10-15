from utils import *
import argparse

import math
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


class Beta_Bayesian:

    def __init__(self, a, b):
        self.a = a  # a is num of success
        self.b = b  # b is num of failure
        self.parameter = a / (a + b)
        self.onDraw = False

    def draw(self):
        if not self.onDraw:
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax.set_title('Online Learner')
            self.onDraw = True
        fig, ax = self.fig, self.ax
        a, b = self.a, self.b
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 10000)
        ax.plot(x, beta.pdf(x, a, b), lw=3, alpha=0.6,
                label='beta (%i,%i)' % (a, b))
        ax.legend(loc='best', frameon=False)
        plt.xlim((0, 1))
        # plt.show()

    @property
    def prior(self):
        return (self.parameter)

    def getPriorPdf(self):
        prior = self.prior
        return prior, beta_pdf(prior, self.a, self.b)

    def getPosteriorPdf(self, evidence):
        new_a = self.a + evidence[1]
        new_b = self.b + evidence[0]
        posterior = new_a / (new_a + new_b)
        return posterior, beta_pdf(posterior, new_a, new_b)

    def calculateLikelihood(self, evidence):
        n_success, n_fail, n_total = evidence[
            1], evidence[0], evidence[0] + evidence[1]
        p = n_success / n_total
        likelihood = nCr(n_total, n_success) * (p**n_success) * ((1 - p)**n_fail)
        return p, likelihood

    def update(self, evidence):
        self.a += evidence[1]
        self.b += evidence[0]
        self.parameter = self.a / (self.a + self.b)
        return self.parameter

    def inference(self, evidence):
        new_a = self.a + evidence[1]
        new_b = self.b + evidence[0]
        posterior = new_a / (new_a + new_b)
        return posterior


def read_data(file_path):
    num_0_list, num_1_list = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            num_0, num_1 = 0, 0
            for x in line:
                if not x.isdigit():
                    continue
                if int(x) == 0:
                    num_0 += 1
                elif int(x) == 1:
                    num_1 += 1
            num_0_list.append(num_0)
            num_1_list.append(num_1)
    return num_0_list, num_1_list

#############################
###  2. Online learning   ###
#############################
# parser = argparse.ArgumentParser()
# parser.add_argument("file_path", default="default")
# parser.add_argument("a", default=3)
# parser.add_argument("b", default=1)
# args = parser.parse_args()

# file_path = args.file_path
# init_a = args.a
# init_b = args.b

file_path = 'test.txt'
init_a = 2
init_b = 2

learner = Beta_Bayesian(init_a, init_b)
learner.draw()
print('Initial Prior is  (a={a},b={b})\n'.format(
    a=learner.a, b=learner.b))
num_0_list, num_1_list = read_data(file_path)
for num_0, num_1 in zip(num_0_list, num_1_list):
    evidence = [num_0, num_1]
    # print('---See evidence of (#0={n_0}, #1={n_1})'.format(n_0=num_0,
    # n_1=num_1) )
    p, likelihood = learner.calculateLikelihood(evidence)
    prior, prior_pdf = learner.getPriorPdf()
    posterior, post_pdf = learner.getPosteriorPdf(evidence)

    print('Get Data (a={num_1},b={num_0}), from (a={a},b={b}) to (a={new_a},b={new_b})\n'
          'Prior      is p={prior:>.4f} (pdf(p)={prior_pdf:>.4f}) \n'
          'Likelihood is p={p:>.4f} (pdf(p)={likelihood:>.4f})\n'
          'Posterior  is p={posterior:>.4f} (pdf(p)={post_pdf:>.4f})\n'.format(
              num_1=num_1, num_0=num_0,
              p=p, likelihood=likelihood,
              prior=prior, prior_pdf=prior_pdf,
              posterior=posterior, post_pdf=post_pdf,
              a=learner.a, b=learner.b,
              new_a=learner.a + num_1, new_b=learner.b + num_0))
    learner.update(evidence)
    learner.draw()

plt.show()
