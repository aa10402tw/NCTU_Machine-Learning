from ChenKW_matrix import ChenKW_Matrix
import argparse
import matplotlib.pyplot as plt


def rLES(xs, ys, n_basis, lambda_):
	'''
	formula:
		equation : x = (ATA + lambda * I)^(-1) * ATb 
		A_ = (ATA + lambda * I)
		b_ = AT * b
		Solve A_*x = b_ with LU to get x 
	'''
	A = ChenKW_Matrix([[ x**i for i in range(n_basis) ] for x in xs])
	b = ChenKW_Matrix([ys]).T()
	I = ChenKW_Matrix.identity(n_basis)
	lambda_I = lambda_ * I 

	A_ = (A.T() * A + lambda_I )
	b_ = A.T() * b
	x = ChenKW_Matrix.LU_solve(A_, b_)

	formula = 'y = ' 
	for i, w in enumerate(x.T()[0]):
		if i > 0 :
			formula += ' + '
		formula += '(%.2f)(x^%i)'%(w, i)
	print('The Formula is : [ %s ]' %(formula) )
	error =  ((A*x - b).T() * (A*x - b))[0][0] + lambda_ * (x.T() * x)[0][0]
	print('The Error is : [ %.2f ]'%error)
	return x


def Newton(xs, ys, n_basis, x0=None, max_iter=100, step_size=1):
	'''
	formula:
		x1 = x0 - (f'(x)/f''(x))
	'''
	if x0 is None:
		x0 = ChenKW_Matrix([[0] for i in range(n_basis)])
	A = ChenKW_Matrix([[ x**i for i in range(n_basis) ] for x in xs])
	b = ChenKW_Matrix([ys]).T()
	x = x0
	for i in range(max_iter):
		gradient =  (2 * A.T() * A * x) -  ( 2 * A.T() * b)
		hession = (2 *A.T() * A) 
		x_new = x - step_size * hession.inv() * gradient
		if(x_new == x):
			print('Converge after %i iteration'%(i+1))
			x = x_new
			break
		x = x_new

	formula = 'y = ' 
	for i, w in enumerate(x.T()[0]):
		if i > 0 :
			formula += ' + '
		formula += '(%.2f)(x^%i)'%(w, i)
	print('The Formula is : [ %s ]' %(formula) )
	error =  ((A*x - b).T() * (A*x - b))[0][0]
	print('The Error is : [ %.2f ]'%error)
	return x

def read_data(file_name):
	xs, ys = [], []
	with open(file_name, 'r') as f:
		lines = f.readlines()
		for line in lines:
			x, y = line.split(',')
			x, y = float(x), float(y)
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
n_basis = 4
lambda_ = 0.1
xs, ys = read_data(file_path)

print('-' * 100)
print("Least Square error with regulation")
rLES(xs, ys, n_basis, lambda_)

print('-' * 100)

print("Newton' method for optimization")
Newton(xs, ys, n_basis)
print('-' * 100)


