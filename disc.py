import numpy as np
from numpy import linalg as LA

## Currently only performing linear discriminant analysis for 2 groups

class disc:

###########
###########

	# constructor for pooled variance
	# A = X1, B = X2
	# abar & bbar are respective mean vectors
	# Sigma is the cov matrix (when assuming pooled) 
	# all these objects are numpy type objects
	def __init__(self, abar, bbar, Sigma1, 
		         A = None, B = None, Sigma2 = None):
		
		self.X1 = A
		self.X2 = B
		self.x1_b = abar
		self.x2_b = bbar

		if Sigma2 is None:
			self.Sp = Sigma1
		else:
			self.Sp1 = Sigma1
			self.Sp2 = Sigma2



###########
	
	# should have functions that can get the objects we need
	# from two data matricies

###########
###########
	
	# get the linear discriminant function,
	# under assumption of pooled covariance
	def getLinearDiscF(self):
		return (self.x1_b - self.x2_b).T * LA.inv(self.Sp)

	# should take the costs and priors I guess
	# for now it takes an observed vector x0.
	# Following 11-18 of the litt. 
	# Function returns TRUE if the observations
	# falls into the first population and
	# FALSE otherwise
	def getRule(self, x0, c1, c2, p1, p2):
		# ruling will differ depending on value of const
		const = (c1/c2)*(p2/p1)
		yhat = self.getLinearDiscF()

	    # if the whole cost prior thing is 1 the ruling looks
	    # a little different than for not 1
		if const == 1:
			m = (self.x1_b-self.x2_b).T * LA.inv(self.Sp) * (self.x1_b + self.x2_b)
			obs = yhat * x0
			# get the ruling (TRUE if 1, false otherwise)
			if obs > m:
				ret = True
			elif obs < m:
				ret = False
		else:
			m = math.log(const)
			obs = yhat - 1/2 * yhat * (self.x1_b + self.x2_b)
			if obs > m:
				ret = True
			else:
				ret = False

		return ret

	### Should have an overloaded versions which
	### doesn't take one observation but rather the data
	
