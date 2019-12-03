import numpy as np
from 
class Linear_Regressor():
	def __init__(self,data):

		pass
	def load_data(self,data_path):

		pass
	def train(self,epocs=0,batch_size=0,optimizer=None,regularization=None):
		ones = np.ones(self.X.shape(0))
		X_bias = np.concatenate((self.X,ones),axis=1)
		if optimizer is None:
			b = np.dot(X_bias.T,self.y)
			A = np.dot(X_bias.T,X_bias)
			self.w = np.dot(np.linalg.pinv(A),b)
		else:
			data_size = len(X[0])
			# initate weights: can delegate intiation for Initiater based on Xavier algorithm or optimizer
			w = np.zeros(X_bias[1])
			for it in range(epocs):
				# shuffle data
				idx_shuf = np.random.permutation(X_bias)
				X_shuffle = X_bias[idx_shuf]
				y_shuffle = y[idx_shuff]
				for batch in range(batch_size):
					if batch +  batch_size > data_size:
						X_batch = X_shuffle[batch:data_size]
						y_batch = y_shuffle[batch:data_size]
					X_batch = X_shuffle[batch:batch+batch_size]
					y_batch = y_shuffle[batch:batch+batch_size]
					grad = self.compute_grad(X_batch,y_batch,w) 
					# update weights
					w = optimizer.update(grad,w)
			loss = self.loss_func(X_bias)
		pass
	def test(self):
		pass
	def loss_func(self,X,y,w,regularization=None):
		if regularization is None:
			self.name = 'Linear Regression'
			loss = 1/2*np.linalg.norm(y-np.dot(X,self.w))
		elif 
		pass
	def compute_grad(self,X,y,w):
		A = np.dot(X,w) - y
		grad  = np.dot(X.T,A)
		return grad
		pass
	def predict(self):
		pass
	@staticmethod
	def shuffle(X):	
		pass