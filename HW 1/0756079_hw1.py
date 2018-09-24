from ChenKW_matrix import ChenKW_Matrix
import argparse
import matplotlib.pyplot as plt


def rLES(xs, ys, n_basis, lambda_):
	'''
	1. Use LU decomposition to find the inverse of (ATA + lambda*I), Gauss-Jordan elimination won't be accepted. A is the design matrix.

	2. Print out the equation of the best fitting line and the error.
	'''
	A = ChenKW_Matrix([[ x**i for i in range(n_basis) ] for x in xs])
	b = ChenKW_Matrix([ys]).T()
	I = ChenKW_Matrix.identity(n_basis)
	lambda_I = I * lambda_
	x = (A.T() * A + lambda_I ).inv() * A.T() * b
	formula = 'y = ' 
	for i, w in enumerate(x.T()[0]):
		if i > 0 :
			formula += '+ '
		formula += '(%.2f)(x^%i) '%(w, i)
	print('The Formula is : [%s]' %(formula) )
	return x


def Newton():
	'''
	1. Print out the equation of the best fitting line and the error, and compare to LSE.
	'''
	pass

def read_data(file_name):
	xs, ys = [], []
	with open(file_name, 'r') as f:
		lines = f.readlines()
		for line in lines:
			x, y = line.split(',')
			x, y = int(x), int(y)
			xs.append(x)
			ys.append(y)
	return xs, ys

# parser = argparse.ArgumentParser()
# parser.add_argument("file_path", default="default")
# parser.add_argument("n_basis", default=3)
# parser.add_argument("lambda_", default=1)
# args = parser.parse_args()
# file_path = args.file_path
# n_basis = args.n_basis
# lambda_ = args.lambda_

file_path = 'test.txt'
n_basis = 3
lambda_ = 0
xs, ys = read_data(file_path)
rLES(xs, ys, n_basis, lambda_)


