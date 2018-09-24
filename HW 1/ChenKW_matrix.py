

''' self-implemented matrix class (Author : NCTU CS 0756079, Chen Kuan-Wen ) '''
class ChenKW_Matrix: 

	def __init__(self, mat):
		if isinstance(mat, ChenKW_Matrix):
			mat = mat.mat
		self.mat = [ mat[i][:] for i in range(len(mat)) ] 
		self.n_row = len(mat)
		self.n_col = len(mat[0])
		self.precision = 2
		self.shape = (self.n_row, self.n_col)

	################################################
	### Operator Overloading make it easy to use ###
	################################################
	def __getitem__(self,key):
		return self.mat[key]

	def __setitem__(self,key,value):
		self.mat[key] = value 

	''' Implementation of matrix multiplication'''
	def __mul__(self, B):
		if isinstance(B, int) or isinstance(B, float):
			return ChenKW_Matrix([[self.mat[i][j] * B for j in range(self.n_col)]  for i in range(self.n_row)])
		if not self.n_col == B.n_row:
			print('Wrong dimension while mat mul') 
			return None
		resultMat = [[sum( [self.mat[i][j] * B.mat[j][k] for j in range(self.n_col)] )  for k in range(B.n_col) ]  for i in range(self.n_row)]
		return ChenKW_Matrix(resultMat)

	''' IImplementation of matrix addition'''
	def __add__(self, B):
		if not ( self.n_row == B.n_row and self.n_col == B.n_col):
			print('Wrong dimension while mat add') 
			return None
		resultMat = [[self.mat[i][j] + B.mat[i][j] for j in range(self.n_col)]  for i in range(self.n_row)]
		return ChenKW_Matrix(resultMat)

	''' Implementation of matrix subtraction'''
	def __sub__(self, B):
		if not ( self.n_row == B.n_row and self.n_col == B.n_col):
			print('Wrong dimension while mat add') 
			return None
		resultMat = [[self.mat[i][j] - B.mat[i][j] for j in range(self.n_col)]  for i in range(self.n_row)]
		return ChenKW_Matrix(resultMat)


	def __len__(self):
		return len(self.mat)

	''' Define what will be printed  when call print(mat) '''
	def __str__(self): 
		max_len = 0
		for i in range(self.n_row):
			for j in range(self.n_col):
				max_len = max(max_len, len(str('%.'+str(self.precision)+'f')%(self.mat[i][j]) ))
		s = '['
		for i in range(self.n_row):
			s += '['
			for j in range(self.n_col):
				s += str('%' + str(max_len) + '.' +str(self.precision)+'f')%(self.mat[i][j]) 
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
	def copy(self):
		''' see copyMatrix(A) in detail '''
		return ChenKW_Matrix.copyMatrix(self)

	def inv(self):
		''' see inverse(A) in detail '''
		return ChenKW_Matrix.inverse(self)

	def T(self):
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

	''' Implementation of Matrix Transpose'''
	@staticmethod
	def concatenate(A1, A2, axis=0):
		A = []
		if(axis == 0):
			for A1_row in A1:
				A.append(A1_row)
			for A2_row in A2:
				A.append(A2_row)
			
		if(axis == 1):
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
	Implementation of LU decomposition (with pivoting), i.e. PA = LU 
		formula:
			PA = LU, let A' = PA, where p i permutation matrix
			u_ij =  a_ij - sum([u_kj * l_ik for k=1 to i-1])
			l_ij = (a_ij - sum([u_{kj}*l_{ik} for k=1 to j-1]) / u_jj
 	'''
	@staticmethod 
	def LUdecomposition(A):
		n = len(A)
		def pivoting_matrix(A):
		    """ Returns the pivoting matrix for A """
		    n = len(A)
		    P = ChenKW_Matrix.identity(n).mat # P =I
		    # Rearrange the P                                                                                                                                                                                             
		    for d in range(n):
		    	max_row_num = max([A[row][d] for row in range(d, n)])
		    	max_row = [A[row][d] for row in range(d, n)].index(max_row_num) + d                                                                                                                                                                        
		    	P[d], P[max_row] = P[max_row], P[d] # Swap the row
		    return P

		L = [[float(i==j) for j in range(n)] for i in range(n)]    # L = I
		U = [[float(0.00) for j in range(n)] for i in range(n)]    # U = O

		P = pivoting_matrix(A)
		PA = (ChenKW_Matrix(P)*ChenKW_Matrix(A)).mat

		for j in range(n):
			for i in range(j+1):
				U[i][j] = PA[i][j] - sum( [U[k][j]*L[i][k] for k in range(i) ] )
			for i in range(j, n):
				L[i][j] = (PA[i][j] - sum([U[k][j]*L[i][k] for k in range(j)]) ) / U[j][j]


		return ChenKW_Matrix(L), ChenKW_Matrix(U), ChenKW_Matrix(P)

	''' Implementation of Inverse using LU '''
	def inverse(A):
		'''
		formula:
		 	 P * A = L * U 
		 	=>  A = inv(P) * L * U
		 	=>  inv(A) = inv(inv(P) * L * U)
		 	=>  inv(A) = inv(U) * inv(L) * P
		'''
		n = len(A)

		# 1. Get the LU decomposition of A, i.e. PA = LU
		L, U, P = ChenKW_Matrix.LUdecomposition(A)

		# 2. Find inverse of L, i.e. inv(L)
		I =  ChenKW_Matrix.identity(n) 						# Create an Identity matrix
		L_I = ChenKW_Matrix.concatenate(L, I, axis=1).mat   # Concate A with I horizatonally
		for d in range(n):
			for row in range(d+1, n):
				m = (L_I[row][d] / L_I[d][d])
				L_I[row] = [ L_I[row][col] - (m * L_I[d][col]) for col in range(len(L_I[d])) ] # row operation(sub)
			L_I[d] = [ L_I[d][col]/L_I[d][d]  for col in range(len(L_I[d])) ] #  scale diagonal to 1 

		L_inv = []
		for L_I_row in L_I:
			I_row = L_I_row[n:]
			L_inv.append(I_row)
		
		# 3. Find inverse of U, i.e. inv(U)
		I =  ChenKW_Matrix.identity(n)
		U_I = ChenKW_Matrix.concatenate(U, I, axis=1).mat
		for d in range(n):
			for row in range(d-1, -1, -1):
				m = (U_I[row][d] / U_I[d][d])
				U_I[row] = [ U_I[row][col] - (m * U_I[d][col]) for col in range(len(U_I[d])) ] # row operation(sub)
			U_I[d] = [ U_I[d][col]/U_I[d][d]  for col in range(len(U_I[d])) ] #  scale diagonal to 1 

		U_inv = []
		for U_I_row in U_I:
			I_row = U_I_row[n:]
			U_inv.append(I_row)

		# 4. Finally, inv(A) = inv(U) * inv(L) * P
		A_inv = ChenKW_Matrix(U_inv) * ChenKW_Matrix(L_inv) * ChenKW_Matrix(P)
		return A_inv
		

if __name__ == '__main__':
	print_precision = 2
	A = ChenKW_Matrix( [[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]] )
	# print(ChenKW_Matrix.concatenate(A,B, axis=1))
	print(A)
	print(A.inv() )
	print(A.inv().inv())
	print(A.inv() * A)
	# print(ChenKW_Matrix.inverse(A_inv))
	# print(L)
	# print(U)
# L, U = LU(A)
# print_matrix(L)
# print_matrix(U)
# print(matmul(A,B))

