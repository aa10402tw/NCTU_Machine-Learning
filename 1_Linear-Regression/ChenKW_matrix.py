
''' self-implemented matrix class (NCTU CS 0756079, Chen Kuan-Wen ) '''
class ChenKW_Matrix: 

	def __init__(self, mat):
		self.mat = [ mat[i][:] for i in range(len(mat)) ] 
		self.n_row = len(mat)
		self.n_col = len(mat[0])
		self.precision = 2
		self.shape = (self.n_row, self.n_col)

	@property
	def T(self):
		''' see transpose(A) in detail'''
		return ChenKW_Matrix.transpose(self)

	@property
	def I(self):
		''' see inverse(A) in detail'''
		return ChenKW_Matrix.inverse(self)

	@property
	def value(self):
		if self.n_row == 1 and self.n_col == 1:
			return self.mat[0][0]
		else:
			return self.mat

	################################################
	### Operator Overloading make it easy to use ###
	################################################
	''' Compare two matrics are equal or not '''
	def __eq__(self, mat):
		if isinstance(mat, ChenKW_Matrix):
			if mat.shape == self.shape:
   				for row_1, row_2 in zip(mat, self):
   					for num_1, num_2 in zip(row_1, row_2):
   						if  not (num_1 - num_2 < 0.0001):
   							return False
   				return True
		else:
			return False

	''' define what will A[i] return '''
	def __getitem__(self,key):
		return self.mat[key]

	''' define what will happen when A[i] = k is called '''
	def __setitem__(self,key,value):
		self.mat[key] = value 

	''' Implementation of matrix multiplication'''
	def __mul__(self, B):
		if isinstance(B, int) or isinstance(B, float):
			return ChenKW_Matrix([[self.mat[i][j] * B for j in range(self.n_col)]  for i in range(self.n_row)])
		if not self.n_col == B.n_row:
			raise Exception('Wrong dimension while mat mul')
		resultMat = [[sum( [self.mat[i][j] * B.mat[j][k] for j in range(self.n_col)] )  for k in range(B.n_col) ]  for i in range(self.n_row)]
		return ChenKW_Matrix(resultMat)
	__rmul__ = __mul__

	''' Implementation of matrix addition'''
	def __add__(self, B):
		if not ( self.n_row == B.n_row and self.n_col == B.n_col):
			raise Exception('Wrong dimension while mat add')
		resultMat = [[self.mat[i][j] + B.mat[i][j] for j in range(self.n_col)]  for i in range(self.n_row)]
		return ChenKW_Matrix(resultMat)
	__radd__ = __add__

	''' Implementation of matrix subtraction'''
	def __sub__(self, B):
		if not ( self.n_row == B.n_row and self.n_col == B.n_col):
			raise Exception('Wrong dimension while mat sub')
		resultMat = [[self.mat[i][j] - B.mat[i][j] for j in range(self.n_col)]  for i in range(self.n_row)]
		return ChenKW_Matrix(resultMat)

	''' define what will return when call len(mat) '''
	def __len__(self):
		return len(self.mat)

	''' Define what will be printed when call print(mat) '''
	def __str__(self): 
		max_len = [0 for col in range(self.n_col)]
		for i in range(self.n_row):
			for j in range(self.n_col):
				max_len[j] = max(max_len[j], len(str('%.'+str(self.precision)+'f')%(self.mat[i][j]) ))
		s = '['
		for i in range(self.n_row):
			s_ = '[' if i==0 else' ['
			s += s_
			for j in range(self.n_col):
				s += str('%' + str(max_len[j]) + '.' +str(self.precision)+'f')%(self.mat[i][j]) 
				if j < self.n_col-1 :
					s += ' '
			if(i >= self.n_row - 1):
				s += ']] \t (%i x %i matrix)\n'%(self.n_row, self.n_col)
			else:
				s += ']\n '
		return s

	######################
	### Useful Methods ###
	######################
	def row_op(self, type, i, j=0, k=0):
		''' see rowOperation() in detail '''
		return ChenKW_Matrix.rowOperation(self, type, i, j, k)

	def copy(self):
		''' see copyMatrix(A) in detail '''
		return ChenKW_Matrix.copyMatrix(self)

	def inv(self):
		''' see inverse(A) in detail '''
		return ChenKW_Matrix.inverse(self)

	def trans(self):
		''' see transpose(A) in detail'''
		return ChenKW_Matrix.transpose(self)

	def LU(self):
		''' see LUdecomposition(A) in detail'''
		return ChenKW_Matrix.LUdecomposition(self)


	######################
	### Static Methods ###
	######################
	''' Implementation of matrix copy (deep copy)'''
	@staticmethod
	def copyMatrix(mat):
		new_mat = [ mat[i][:] for i in range(len(mat)) ] 
		return ChenKW_Matrix(new_mat)

	''' Implementation of Matrix Transpose'''
	@staticmethod
	def identity(n):
		return ChenKW_Matrix( [[float(i==j) for j in range(n)] for i in range(n)] )

	''' Implementation of matrix row operation'''
	@staticmethod
	def rowOperation(mat, type, i, j=0, k=0):
		if type == 1:  # mult row_i for k times
			mat[i] = [ mat[i][col] * k for col in range(len(mat[i]))]
			return ChenKW_Matrix(mat)
		elif type == 2: # row_i exchange with row_j
			mat[i], mat[j] = mat[j], mat[i]
			return ChenKW_Matrix(mat)
		elif type == 3: # Add row_i * k to row_j
			mat[j] = [ mat[j][col] + k * mat[i][col] for col in range(len(mat[i]))]
			return ChenKW_Matrix(mat)

	''' Implementation of Matrix Transpose'''
	@staticmethod
	def concatenate(A1, A2, axis=0):
		A = []
		if(axis == 0): # Vertically concatenate
			for A1_row in A1:
				A.append(A1_row)
			for A2_row in A2:
				A.append(A2_row)
			
		elif(axis == 1): # Horizationally concatenate
			for A1_row, A2_row in zip(A1, A2):
				A.append(A1_row + A2_row) # List append
		return ChenKW_Matrix(A)

	''' 
	Implementation of Matrix Transpose
		formula: B = A^T, where b_ij = a_ji
	'''
	@staticmethod 
	def transpose(A):
		if isinstance(A, ChenKW_Matrix):
			A = A.mat
		A_t  = list(map(list, zip(*A))) # unpack list and zip it, then map a list through it
		return ChenKW_Matrix(A_t)

	''' 
	Implementation of  using LU decomposition to solve linear equation
		formula:
			Ax = b, and A = LU
			(LU)x = L(Ux) = Ly = b, solve y
			then solve Ux = y
 	'''
	@staticmethod 
	def LU_solve(A, b):
		L, U = ChenKW_Matrix.LUdecomposition(A)
		y = U * ChenKW_Matrix(b)

		# Solve L * y = b
		L_b = ChenKW_Matrix.concatenate(L, b, axis=1)
		for d in range(L_b.n_row):
			for row in range(d+1, L_b.n_row):
				m = -1 * (L_b[row][d] / L_b[d][d])
				L_b = L_b.row_op(type=3, i=d, j=row, k=m)
		y = ChenKW_Matrix([[row[-1] for row in L_b]]).T

		# solve Ux = y
		U_y = ChenKW_Matrix.concatenate(U, y, axis=1)
		for d in range(U_y.n_row):
			U_y = U_y.row_op(type=1, i=d, k=(1/U_y[d][d]))
			for row in range(d-1, -1, -1):
				m = -1 * (U_y[row][d] / U_y[d][d])
				U_y = U_y.row_op(type=3, i=d, j=row, k=m)
		x = ChenKW_Matrix([[row[-1] for row in U_y]]).T 
		return x

	''' 
	Implementation of LU decomposition (without pivoting), i.e. A = LU 
		formula:
			u_ij =  a_ij - sum([u_kj * l_ik for k=1 to i-1])
			l_ij = (a_ij - sum([u_kj * l_ik for k=1 to j-1]) / u_jj
 	'''
	@staticmethod 
	def LUdecomposition(A):
		n = len(A)
		L = [[float(i==j) for j in range(n)] for i in range(n)]    # L = I
		U = [[float(0.00) for j in range(n)] for i in range(n)]    # U = O

		for j in range(n):
			for i in range(j+1):
				U[i][j] = A[i][j] - sum( [U[k][j]*L[i][k] for k in range(i) ] )
			for i in range(j, n):
				L[i][j] = (A[i][j] - sum([U[k][j]*L[i][k] for k in range(j)]) ) / U[j][j]
		return ChenKW_Matrix(L), ChenKW_Matrix(U)

	
	''' 
	Implementation of Matrix Inverse using Gaussian-Jordan elimination
		formula:
			[ A | I ] --> [ I | inv(A) ]  where '-->' represent row operations 
 	'''
	@staticmethod
	def inverse(A):
		A = ChenKW_Matrix.copyMatrix(A)
		if not A.n_row == A.n_col:
			raise Exception('Not Square matrix has no inverse.')
		n = A.n_row
		I = ChenKW_Matrix.identity(n)
		A_I = ChenKW_Matrix.concatenate(A, I, axis=1)
		for d in range(n):
			# Make Sure diagonal is non-zero
			for row in range(d, n):
				if not A_I[row][d] == 0:
					A_I = A_I.row_op(type=2, i=d, j=row) # exchange non-zero row to diagonal
					break
			if A_I[d][d] == 0:
				raise Exception('Singular matrix has no inverse.')

			# Do row operations to make the elements of column is 1 if diagonal, 0 otherwise 
			A_I = A_I.row_op(type=1, i=d, k=(1/A_I[d][d])) 
			for row in range(n):
				if d==row: continue
				m = -1*(A_I[row][d]/A_I[d][d])
				A_I = A_I.row_op(type=3, i=d, j=row, k=m)
		A_inv = ChenKW_Matrix([row[n:] for row in A_I])
		return A_inv


		

if __name__ == '__main__':
	print_precision = 2
	A = ChenKW_Matrix( [[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]] )
	print(A)
	print(A.inv())
	print(A.inv() * A)
	print(A.inv().inv())
	b = ChenKW_Matrix( [[7, 3, -1, 2]] )
	print(b)
	print(b.T)


