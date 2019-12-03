import numpy as np
def bincount2D(A,weight=None,axis=0,minlength=0):
	if weight is None:
		return np.apply_along_axis(lambda x: np.bincount(x,minlength=minlength),axis=axis,A)
	elif A.shape!==weight.shape:
		return TypeError
	else:
		order = {0:(A,weight),1:(A.T,weight.T)}
		m = A.max()+1
		n = A.shape[axis]
		A_flat = order[axis][0] + m*np.arange(n)
		weight_flat = order[axis][1]
		out = np.bincount(A_flat.ravel(),weight_flat,minlength=m*n)
	pass