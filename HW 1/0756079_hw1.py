from ChenKW_matrix import ChenKW_Matrix
import argparse


def LES():
	'''
	1. Use LU decomposition to find the inverse of (ATA + lambda*I), Gauss-Jordan elimination won't be accepted. A is the design matrix.

	2. Print out the equation of the best fitting line and the error.
	'''
	pass

def Newton():
	'''
	1. Print out the equation of the best fitting line and the error, and compare to LSE.
	'''
	pass

def read_data():
	'''
	each row represents a data point (common seperated: x,y): 

        1,12

        122,34

        -12,323
    '''
	pass

parser = argparse.ArgumentParser()
parser.add_argument("file_path", default="default")
parser.add_argument("n_basis", default=3)
parser.add_argument("lambda_", default=1)

args = parser.parse_args()
print(args.file_path)
print(args.n_basis)
print(args.lambda_)

