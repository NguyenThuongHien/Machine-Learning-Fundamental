import numpy as np
class SGD():
	def __init__(self,learning_rate,momentum):
		self.lr = learning_rate
		self.momentum = momentum
		self.v = None
		pass
	def update(self,grad,w):
		if self.v is None:
			self.v = np.zeros_like(w)
		self.v = self.momentum*self.v + self.lr*grad
		return w - self.v
		passb 
	def compute_grad(w,func):
		eps = 1e-4
		return (func(w+eps) -func(w-eps))/(2*eps)
		pass