import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

from utils import *


class NaiveBayes_Discrete():

    def __init__(self, num_bins=32, min_count=1, min_=0, max_=256):
        self.num_bins = num_bins
        self.min_count = min_count
        self.min = min_
        self.max = max_
        self.interval = (self.max - self.min) // self.num_bins

    def fit(self, X, Y):
        N, D = X.shape
        self.labels = set(Y)
        self.bins = dict()
        self.priors = dict()
        for c in self.labels:
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
            self.bins[c] = np.zeros((D, self.num_bins)).astype(np.int32)
            x_c = X[Y == c]
            f_c = np.transpose(x_c // self.interval, (1, 0))
            for i, fb in enumerate(f_c):
                for b in range(self.num_bins):
                    count = len(fb[fb == b])
                    self.bins[c][i][b] = max(self.min_count, count)

    def eval(self, X, Y):
        y_pred = self.predict(X)
        acc = (y_pred == Y_test).sum() / len(Y)
        return acc

    def predict(self, X):
        N, D = X.shape
        y_pred = []
        for x in X:
            log_probs = [0 for x in range(len(self.labels))]
            for c in self.labels:
                log_probs[c] = math.log(self.priors[c])
                xb = x[:] // self.interval
                for f, b in enumerate(xb):
                    log_prob = math.log(
                        self.bins[c][f][int(b)] / self.bins[c][f].sum())
                    log_probs[c] += log_prob
            pred = argmax(log_probs)
            y_pred.append(pred)
        return y_pred


class NaiveBayes_Continuous():

    def fit(self, X, Y):
        N, D = X.shape
        self.labels = set(Y)
        self.gaussian = dict()
        self.priors = dict()
        for c in self.labels:
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
            self.gaussian[c] = [Gaussian() for f in range(D)]
            x_c = X[Y == c]
            for x in x_c:
                for f in range(D):
                    self.gaussian[c][f].update(
                        x[f] / 255.0)  # normalize to (0~1)

    def eval(self, X, Y):
        y_pred = self.predict(X)
        acc = (y_pred == Y_test).sum() / len(Y)
        return acc

    def predict(self, X):
        N, D = X.shape
        y_pred = []
        for x in X:
            log_probs = [0 for x in range(len(self.labels))]
            for c in self.labels:
                log_probs[c] = math.log(self.priors[c])
                for f, b in enumerate(x):
                    log_prob = self.gaussian[c][f].logpdf(x[f] / 255.0)
                    log_probs[c] += log_prob
            pred = argmax(log_probs)
            y_pred.append(pred)
        return y_pred


#  Naive Bayes classifier
# 　0 for discrete mode and 1 for continuous mode
MODES = ['0', 'Discrete', '1', 'Continuous']

# Get user argument
parser = argparse.ArgumentParser()
parser.add_argument
parser.add_argument("-mode", "--mode-option", help="Select mode",
                    dest="mode", default="0", choices=(tuple(MODES)))
args = parser.parse_args()
MODE = args.mode
if MODE == '0':
    MODE = 'Discrete'
elif MODE == '1':
    MODE = 'Continuous'


# # Select model based on mode
print('Use model NaiveBayes_%s' % (MODE))
model = NaiveBayes_Discrete() if MODE == 'Discrete' else NaiveBayes_Continuous()

# Load mnist data
print('Load Data...')
imgs_train, labels_train = load_mnist(train=True)
imgs_test, labels_test = load_mnist(train=False)

# Change Image to Feature Vector
X_train, Y_train = img_to_vector(imgs_train), labels_train
X_test, Y_test = img_to_vector(imgs_test), labels_test

# Fit training data
print('Fitting on training data...')
model.fit(X_train, Y_train)

# Test on test set
print('Test on test data...')
acc = model.eval(X_test, Y_test)
print("Test Acc:", acc)

# Predict on random selected images and show results
imgs_display = imgs_train[np.random.choice(imgs_train.shape[0], 10), :]  # 選擇 random 的那 10 列，以及所有的行
X_display = img_to_vector(imgs_display)
Y_display = model.predict(X_display)
for i in range(10):
    img = imgs_display[i]
    title = 'Prediction : %s' % (str(Y_display[i]))
    plt.subplot(2, 5, i + 1), plt.imshow(img, cmap='gray')
    plt.title(title), plt.yticks([]), plt.xticks([])
plt.show()
