# from ChenKW_matrix import ChenKW_Matrix
import argparse
import numpy as np
import math

from utils import *

#  Naive Bayes classifier 
MODES = ['Discrete', 'Continuous'] #　discrete mode and continuous mode

# Get user argument
parser = argparse.ArgumentParser()
parser.add_argument
parser.add_argument("-mode", "--mode-option", help="Select mode", dest="mode", default="Continuous", choices=(tuple(MODES)))
args = parser.parse_args()
MODE = args.mode
# MODE = 'Discrete'

class NaiveBayes_Continuous():

	def __init__(self):
		pass

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
					self.gaussian[c][f].update(x[f]/255.0) # normalize to (0~1)

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
					log_prob = self.gaussian[c][f].logpdf(x[f]/255.0)
					log_probs[c] += log_prob
			pred = argmax(log_probs)
			y_pred.append(pred)
		return y_pred



class NaiveBayes_Discrete():
	def __init__(self, num_bins=32, min_count=1, min_=0, max_=256):
		self.num_bins = num_bins
		self.min_count = min_count
		self.min = min_
		self.max = max_
		self.interval = (self.max-self.min) // self.num_bins

	def fit(self, X, Y):
		N, D = X.shape
		self.labels = set(Y)
		self.bins = dict()
		self.priors = dict()
		for c in self.labels:
			self.priors[c] = float(len(Y[Y == c])) / len(Y)
			self.bins[c] = np.zeros( (D, self.num_bins) ).astype(np.int32)
			x_c = X[Y == c]
			f_c = np.transpose( x_c // self.interval, (1,0))
			for i, fb in enumerate(f_c):
				for b in range(self.num_bins):
					count = len(fb[fb==b])
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
					log_prob = math.log(self.bins[c][f][int(b)] / self.bins[c][f].sum())
					log_probs[c] += log_prob
			pred = argmax(log_probs)
			y_pred.append(pred)
		return y_pred



# Load mnist Data
print('Load Data...')
imgs_train, labels_train = load_mnist(train=True)
imgs_test, labels_test = load_mnist(train=False)

# Change Image to Feature Vector
X_train, Y_train = img_to_vector(imgs_train), labels_train
X_test, Y_test  = img_to_vector(imgs_test), labels_test

if MODE == 'Discrete':

	model = NaiveBayes_Discrete()
	model.fit(X_train, Y_train)
	acc = model.eval(X_test, Y_test)
	print("Test Acc:", acc)

	# class_count = [0 for x in range(n_class)]
	# features = [ [ [Feature_Bins(32, id='(class=%i, pos=%s,%s)'%(c, row, col), min_count=1)  for col in range(n_col)] for row in range(n_row) ]for c in range(n_class)]
	
	# # Fitting on training set
	# print('Fitting on training set...')
	# for i, (img, label) in enumerate(zip(imgs_train, labels_train)):
	# 	class_ = int(label)
	# 	class_count[class_] += 1
	# 	for row in range(n_row):
	# 		for col in range(n_col):
	# 			value = img[row][col]
	# 			features[class_][row][col].to_bin(value)

	# # Test
	# print('Testing on test set...')
	# correct, total = 0, 0
	# for img, label in zip(imgs_test, labels_test):
	# 	label = int(label)
	# 	# Compute prob of each class
	# 	probs = [class_count[class_] / sum(class_count) for class_ in range(n_class)]
	# 	log_probs = [math.log(prob) for prob in probs]
	# 	for class_ in range(n_class):
	# 		for row in range(n_row):
	# 			for col in range(n_col):
	# 				value = img[row][col]
	# 				bin_num = features[class_][row][col].get_bin_num(value)
	# 				prob = features[class_][row][col].get_count(bin_num) / features[class_][row][col].total_count()
	# 				log_probs[class_] += math.log(prob) 

	# 	# Argmax
	# 	label_predict = argmax(log_probs)

	# 	# Test if correct or not
	# 	total += 1
	# 	if label_predict == label:
	# 		correct += 1

	# print('Test Accuracy: %.2f (%i/%i)'%(correct/total, correct, total) )
		

elif MODE == 'Continuous':
	model = NaiveBayes_Continuous()
	model.fit(X_train, Y_train)
	acc = model.eval(X_test, Y_test)
	print("Test Acc:", acc)


	# class_count = [0 for x in range(n_class)]
	# features = [ [ [Feature_Gaussian(id='(class=%i, pos=%s,%s)'%(c, row, col))  for col in range(n_col)] for row in range(n_row) ]for c in range(n_class)]
	# print('Fitting on training set...')
	# for i, (img, label) in enumerate(zip(imgs_train, labels_train)):
	# 	class_ = int(label)
	# 	class_count[class_] += 1
	# 	for row in range(n_row):
	# 		for col in range(n_col):
	# 			value = img[row][col]
	# 			features[class_][row][col].update(value/255.0)

	# # Test
	# print('Testing on test set...')
	# correct, total = 0, 0
	# for img, label in zip(imgs_test, labels_test):
	# 	label = int(label)
	# 	# Compute Distance to each class
	# 	probs = [ (class_count[class_]/sum(class_count)) for class_ in range(n_class)]
	# 	log_probs = [math.log(prob) for prob in probs]
	# 	for class_ in range(n_class):
	# 		for row in range(n_row):
	# 			for col in range(n_col):
	# 				value = img[row][col]
	# 				log_prob = features[class_][row][col].log_pdf(value/255.0)
	# 				# print(prob, end=',')
	# 				log_probs[class_] += log_prob 

	# 			# Argmax
	# 	label_predict = argmax(log_probs)
	# 	#print('Predict label: {label} ({prob:.2f}%)'.format(label=label_predict, prob=100*probs[label_predict]) )

	# 	# Test if correct or not
	# 	total += 1
	# 	if label_predict == label:
	# 		correct += 1

	# print('Test Accuracy: %.2f (%i/%i)'%(correct/total, correct, total) )





