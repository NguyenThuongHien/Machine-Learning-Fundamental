# Kmeans ++ algorithm:
# 	init a c1 chosen uniformly at random from dataset
# 	init ci chosen randomly from X with probability D(x)^2/sum(D(x)^2)
# 	repeat step 2 until reach at k centers
import numpy as np
from scipy.spatial.distance import cdist
def get_min_dist(centers,X):
	return np.min(cdist(centers,X),axis=0)
	pass
def get_distribution_choice(Dx):
	sum_Dx=np.sum(Dx)
	return Dx/sum_Dx
	pass
def init_center(k,X):
	centers=np.zeros((k,X.shape[1]))
	randint=np.random.randint(k,size=1)[0]
	centers[0]=X[randint]
	# Can improve by omitting chosen points in dataset
	for i in range(1,k):
		now_centers=centers[:i]
		Dx=get_min_dist(now_centers,X)
		distribution=get_distribution_choice(Dx)
		centers[i]=X[np.random.choice(X.shape[0],1,p=distribution)]
	return centers
	pass
# Proceed similarly traditional kmeans