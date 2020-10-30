from ChenKW_matrix import ChenKW_Matrix
import argparse


def get_formula(w):
	formula = 'y = '  
	for i, w_ in enumerate(w.T[0]):
		if i > 0 :
			formula += ' + '
		formula += '(%.2f)(x^%i)'%(w_, i)
	return formula

def rLSEerror(xs, ys, w, lambda_=0):
	A = ChenKW_Matrix([[ x**i for i in range(n_basis) ] for x in xs])
	b = ChenKW_Matrix([ys]).T
	error = ((A*w-b).T * (A*w-b)) + lambda_*(w.T*w)
	error = error.value
	return error

def rLES(xs, ys, n_basis, lambda_):
	'''
	formula:
		equation : x = (A_T * A + lambda * I)^(-1) * A_T * b 
		A_ = (A_T * A + lambda * I)
		b_ = A_T * b
		Solve A_*x = b_ with LU to get x 
	'''
	A = ChenKW_Matrix([[ x**i for i in range(n_basis) ] for x in xs])
	b = ChenKW_Matrix([ys]).T
	I = ChenKW_Matrix.identity(n_basis)
	lambda_I = lambda_ * I 

	A_ = (A.T * A + lambda_I )
	b_ = A.T * b
	x = ChenKW_Matrix.LU_solve(A_, b_)
	return x


def Newton(xs, ys, n_basis, x0=None, max_iter=100, step_size=1):
	'''
	formula:
		equation : x_new = x - step_size*(f'(x)/f''(x)) 
			or x_new = x - step_size*inv(Hession)*gradient
			where Hession = 2*A^T*A, gradient = 2*A^T*A*x - 2*A^T*b
		let y = inv(Hession)*gradient, A' = Hession, and b' = gradient
		we can get y by solving A'y = b' using LU decompsition
	'''

	if x0 is None:
		x0 = ChenKW_Matrix([[0] for i in range(n_basis)])
	A = ChenKW_Matrix([[ x**i for i in range(n_basis) ] for x in xs])
	b = ChenKW_Matrix([ys]).T
	x = x0
	for i in range(max_iter):
		A_ = hession = (2 *A.T * A) 
		b_ = gradient =  (2 * A.T * A * x) -  ( 2 * A.T * b)
		step_dir = y = ChenKW_Matrix.LU_solve(A_, b_)
		x_new = x - (step_size * step_dir)
		if(x_new == x):
			print('Converge after %i iteration'%(i))
			x = x_new
			break
		x = x_new
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

parser = argparse.ArgumentParser()
parser.add_argument("file_path", default="default")
parser.add_argument("n_basis", default=3)
parser.add_argument("lambda_", default=1)
args = parser.parse_args()
file_path = args.file_path
n_basis = int(args.n_basis)
lambda_ = float(args.lambda_)

# file_path = 'test.txt'
# n_basis = 3
# lambda_ = 0

xs, ys = read_data(file_path)

print("#---- Least Square error with regularization ----#")
w = rLES(xs, ys, n_basis, lambda_)
print('Formula of rLSE is [ {formula} ]'.format(formula=get_formula(w)))
print('Error of rLSE is [ {error:.4f} ]'.format(error=rLSEerror(xs, ys, w, lambda_)))

print()

print("#---- Newton' method for optimization ----#")
w = Newton(xs, ys, n_basis)
print('Formula of Newton is [ {formula} ]'.format(formula=get_formula(w)))
print('Error of Newton is [ {error:.4f} ]'.format(error=rLSEerror(xs, ys, w, lambda_)))


