# knn for classification
# 	find k nearest neighbor
# 	get maximal of class have highest point in k neighbor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 

from scipy.spatial.distance import cdist
import numpy as np
class KNN():
	def __init__(self,data_path,k,p):
		self.path = data_path
		self.k = k
		self.p = p
		iris = load_iris()
		self.data = iris.data
		self.target = iris.target
		self.data = self.__normalize(self.data)
		self.X_train,self.X_test,self.y_train,self.y_test=tts(self.data,self.target,test_size=50)
		pass
	def __load_data(self):
		pass
	def __normalize(self,data):
		return(data/np.amax(data,axis=0))
		pass
	def __compute_distance(self,x):
		distance = cdist(self.X_train,x)
		return distance
		pass
	def __get_nearest_neighbor(self,distance):
		return np.argsort(distance,axis=0)[:self.k,:]
		pass
	def __weight_func(self,distance,alpha=0.5):
		# name: '-1':1/al+d,'-2':1/al+d^2,e:e^-(d/al)^2
		weight = 0.0
		if self.p == 1:
			return 1/(alpha+distance)
		elif self.p == 2:
			return 1/(alpha+np.pow(distance,-2))
		elif self.p == 'e':
			return np.exp(-(distance/alpha)**2)
		return 'valid name'
		pass
	def __classify(self,x):
		distance = self.__compute_distance(x)
		# idx_row is row's index of postion have maximum 
		idx_row = self.__get_nearest_neighbor(distance)

		idx_col = np.tile(np.arange(idx_row.shape[1]),(self.k,1))
		k_nearests = (idx_row,idx_col)
		distance_nn = distance[k_nearests]
		weights = self.__weight_func(distance_nn)

		labels = np.take(self.y_train,idx_row)
		m=labels.max()+1
		# vectorize to calculate bincount with weights
		n = labels.shape[1]
		labels_flat = labels + m*np.arange(n)
		labels_flat = labels_flat.flatten('F')
		weights_flat = weights.flatten('F')
		# handle with k = 1 with shape error
		out = np.bincount(labels_flat,weights_flat,minlength=m*n)
		points = out.reshape(n,-1).T

		class_of_x = np.argmax(points,axis=0)
		return class_of_x
		pass
	def __regression(self,x):
		distance = self.__compute_distance(x)
		idx_row = self.__get_nearest_neighbor(distance)
		idx_col = np.tile(np.arange(distance.shape[1]),(self.k,1))
		k_nearests = (idx_row,idx_col)

		labels = np.take(self.y_train,idx_row)
		distance_nn = distance[k_nearest]
		weights = self.__weight_func(distance_nn)

		value = np.sum(distance_nn*weights,axis=0)/np.sum(sum_weight,axis=0)
		return value
		pass
	def predict(self,x):
		pass
	def test(self):
		y_pred = self.__classify(self.X_test)
		print(y_pred)
		return accuracy_score(self.y_test,y_pred)
		pass
if __name__ == '__main__':
	k=1
	knn = KNN('',k,'e')
	print("Accuracy with k:{} = {} %".format(k,knn.test()*100))
# drawbacks of knn:
	# choose k?
	# efficient in computation k nearest neighbor
	# choose metric in computation distance
	# sensitive to noise
# solution
	# choose k: use elbow method, PCD,SVD to tranform data