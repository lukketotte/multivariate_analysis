import numpy as np
from numpy import linalg as LA

class cca:

	# constructor
	# a = upper limit for R11
	# b = upper limit for R22
	def __init__(self, A, a, b):
		self.R11 = A[0:a, 0:a]
		self.R22 = A[a:b, a:b]
		self.R12 = A[0:a, a:b]
		self.R21 = self.R12.T

	# negative square root of matrix
	def neg_sqrt_mat(self, A):
		w,v = LA.eig(A)
		ret = v * np.diag(np.sqrt(1/w)) * v.T
		return ret

	#get canonical correlations
	def rho(self):
		R11_s = self.neg_sqrt_mat(self.R11)
		w, v = LA.eig(R11_s * self.R12 * LA.inv(self.R22) * self.R21 * R11_s)
		# has to be in descending order,
		# [::-1] flips the vector
		return np.sort(np.sqrt(w))[::-1]

	# get canonical covariates for X1
	# a = number of covariates to return
	def getU(self, a):
		R11_s = self.neg_sqrt_mat(self.R11)
		w, v = LA.eig(R11_s * self.R12 * LA.inv(self.R22) * self.R21 * R11_s)
		# standardize and return
		# note: v[:, 0:a] is A_z
		return v[:, 0:a].T * R11_s

	# get canonical covariates for X2
	# a = number of covariates to return
	def getV(self, a):
		R22_s = self.neg_sqrt_mat(self.R22)
		w, v = LA.eig(R22_s * self.R21 * LA.inv(self.R11) * self.R12 * R22_s)
		# standardize and return
		# note: v[:, 0:a] is B_z
		return v[:, 0:a].T * R22_s   

	#### Get full A & B matricies ####

	def getA(self):
		R11_s = self.neg_sqrt_mat(self.R11)
		w, v = LA.eig(R11_s * self.R12 * LA.inv(self.R22) * self.R21 * R11_s)
		return v.T * R11_s

	def getB(self):
		R22_s = self.neg_sqrt_mat(self.R22)
		w, v = LA.eig(R22_s * self.R21 * LA.inv(self.R11) * self.R12 * R22_s)
		return v.T * R22_s

	#### Correlations beteween variates and variables ####

	# correlation between U (matrix) and x1, first set
	def getRUx1(self):
		A = self.getA()
		dim = A.shape  # assume it's square
		# set dimension of np.identity()
		return A * self.R11 * np.identity(dim[1])

	# correlation between V (matrix) and x2, second set
	def getRVx2(self):
		B = self.getB()
		dim = B.shape
		return B * self.R22 * np.identity(dim[1])

	def getRUx2(self):
		A = self.getA()
		dim = self.R22.shape
		return A * self.R12 * np.identity(dim[1])

	def getRVx1(self):
		B = self.getB()
		dim = self.R11.shape
		return B * self.R21 * np.identity(dim[1])

	### last part: proportion of explained variance ###
	# U explained in the first set. Not sure if working
	def getPropU(self, a):
		# take the inverse of A
		A = LA.inv(self.getA())
		A = np.square(A)
		# we need to slice it up so it corresponds 
		# too how many canonical variates the user 
		# has
		A = A[: , 0:a]
		# return A
		return np.trace(A)/np.trace(self.R11)