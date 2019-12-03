
class L1():
	def __init__(self,lbd):
		self._lambda = lbd
		pass
	def func(self,w):
		return self._lambda*np.linalg.norm(w,1)
		pass
	def grad(self):
		self._lambda*np.sign(w)
		pass

class L2():
	def __init__(self,lbd):
		self._lambda = lbd
		pass
	def func(self,w):
		return 0.5*self._lambda*np.linalg.norm(w)
		pass
	def grad(self):
		self._lambda*np.sign(w)
		pass
