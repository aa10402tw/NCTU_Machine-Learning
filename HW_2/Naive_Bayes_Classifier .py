# from ChenKW_matrix import ChenKW_Matrix
import argparse
import numpy as np

from utils import *

#  Naive Bayes classifier 
MODES = ['Discrete', 'Continuous'] #　discrete mode and continuous mode

# Load Mnist Data
print('Load Data...')
imgs_train, labels_train = load_mnist(train=True)
imgs_test, labels_test = load_mnist(train=False)

# Get user argument
parser = argparse.ArgumentParser()
parser.add_argument
parser.add_argument("-mode", "--model-option", help="Select mode", dest="mode", default="Discrete", choices=(tuple(MODES)))
args = parser.parse_args()
MODE = args.mode

n_row, n_col = 28, 28
n_class = 10
n_img = 60000

if MODE == 'Discrete':
	class_count = [0 for x in range(n_class)]
	features = [ [ [Feature_Bins(32, id='(class=%i, pos=%s,%s)'%(c, row, col), min_count=100)  for col in range(n_col)] for row in range(n_row) ]for c in range(n_class)]
	
	# Fitting on training set
	print('Fitting on training set...')
	for i, (img, label) in enumerate(zip(imgs_train, labels_train)):
		class_ = int(label)
		class_count[class_] += 1
		for row in range(n_row):
			for col in range(n_col):
				value = img[row][col]
				features[class_][row][col].to_bin(value)
				# features[class_][row][col].pseudocount(min_count=100)
		if i % 1000 ==0:
			print(i, end=',')
	print(features[0][0][0])

	# Test
	print('Testing on test set...')
	correct, total = 0, 0
	for img, label in zip(imgs_test, labels_test):
		label = int(label)
		# print('Correct Label :', int(label))
		# Compute prob of each class
		probs = [0 for x in range(n_class)]
		for class_ in range(n_class):
			probs[class_] = class_count[class_] / sum(class_count)
			for row in range(n_row):
				for col in range(n_col):
					value = img[row][col]
					bin_num = features[class_][row][col].get_bin_num(value)
					probs[class_] *= features[class_][row][col].get_count(bin_num) / features[class_][row][col].total_count()
					#print('count:', features[class_][row][col].get_count(bin_num), features[class_][row][col].total_count())
		# print('Origin probs')
		# print_probs(probs)

		# Normalize
		probs_sum = sum(probs) 
		print(probs_sum)
		if probs_sum > 0:
			probs = [ prob/probs_sum for prob in probs ]
			# print('Normalized prob')
			# print_probs(probs)

		# Argmax
		label_predict = argmax(probs)
		#print('Predict label: {label} ({prob:.2f}%)'.format(label=label_predict, prob=100*probs[label_predict]) )

		# Test if correct or not
		total += 1
		if label_predict == label:
			correct += 1
		if total % 1000 == 0:
			print(correct/total)

	print('Test Accuracy: %.2f (%i/%i)'%(correct/total, correct, total) )
		





elif MODE == 'Continuous':

	'''
	Trainning 
	For each class = 0..9
		get all x’s (images) for the class
		save the mean and covariance of those x’s with the class
	'''

	'''
	Testing
	Given a test point x:
	    Calculate the probability that x is in each class c
	    Return the class c that yields the highest posterior probability
	'''

	pass





