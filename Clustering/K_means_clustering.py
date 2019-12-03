# Thuật toán Kmeans
# Khởi tạo tâm cụm
# Phân bố lại cụm cho từng điểm
# 	+Tính khoảng cách của mỗi điểm tới từng tâm cụm
# 	+Tìm min khoảng cách tới từ điểm đó tới tâm cụm
# Tính lại tâm cụm
# Trở lại bước 3

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Initiate data point for practicing
# Parameters:
# 	centroids: list coordinate of all centroids
# 	ndpcl: number data point per cluster
def init_data(centroids,ndpc):
	data_set=[]
	R=random.randint(1,4)
	for cent in centroids:
		point_set=[]
		for i in range(ndpc):
			a=random.random()*2*math.pi
			r=R*math.sqrt(random.random())
			point=[cent[0]+r*math.cos(a),cent[1]+r*math.sin(a)]
			point_set.append(point)
		data_set.extend(point_set)
	return np.array(data_set)
def visualize(centroids,data_set):
	colors=['blue','green','red','yellow','orange']
	for i,cluster in enumerate(data_set):
		color_data=colors[i%len(colors)]
		color_cen=colors[(i+1)%len(colors)]
		plt.plot(cluster[:,0],cluster[:,1],'o',marker='*',markersize=2,color=color_data,label='cluster '+str(i))
		plt.plot(centroids[:,0],centroids[:,1],'*',marker='s',markersize=5,color='black')
	plt.legend()
	plt.show()
	pass
#Initiate randomly center of each cluster
#Parameters:
#	k: number of centroid
#	data_set: set of all points 
def init_centroid(data_set,k):
	centroids=np.random.choice(data_set.shape[0],k)
	return data_set[centroids]
	pass
# Compute and return nearest centroid for each point in data set
# Parameter:
# point: a point in data set
# centroids: list all centers

def get_neareast(data_set,centroids):
	from scipy.spatial.distance import cdist
	dist=cdist(centroids,data_set)
	return np.argmin(dist,axis=0)
	pass
def update_centroid(cluster_label,data_set,k):
	centroids=np.zeros((k,data_set.shape[1]))
	for i in range(k):
		Xi=data_set[np.where(cluster_label==i)]
		centroids[i]=np.mean(Xi,axis=0)
	return centroids
	pass
def has_stopped(old_cluster,new_cluster):
	return np.array_equal(old_cluster,new_cluster)
	pass
def kmeans(data_set,k,no_iter=None):
	centroids=init_centroid(data_set,k)
	clus_label=[]

	centroid_set=[]
	label_set=[]
	if no_iter==None:
		while True:
			clus_label=get_neareast(data_set,centroids)
			new_centroids=update_centroid(clus_label,data_set,k)
			if has_stopped(centroids,new_centroids):
				break
			centroids=new_centroids
			centroid_set.append(centroid_set)
			label_set.append(clus_label)
		return clus_label,centroids,centroid_set,label_set
	else:
		for i in range(no_iter):
			clus_label=get_neareast(data_set,centroids)
			new_centroids=update_centroid(clus_label,data_set,k)
			if has_stopped(centroids,new_centroids):
				break
			centroids=new_centroids
			centroid_set.append(centroid_set)
			label_set.append(clus_label)
		return clus_label,centroids,centroid_set,label_set
	pass
def animate_kmeans(centroids,data_set,k,no_iter=None):
	# centroids=init_centroid(data_set,k)
	clus_label=[]
	it=0
	while True:
		it+=1
		clus_label=get_neareast(data_set,centroids)
		new_centroids=update_centroid(clus_label,data_set,k)
		if has_stopped(centroids,new_centroids):
			break
		centroids=new_centroids
		yield centroids,clus_label
	pass
def get_result_kmeans(generator):
	while 1:
		centroids,labels=[],[]
		try:
			centroids,labels=next(generator)
		except Exception as e:
			return centroids,labels
	pass
def divide_data(data_set,label,centroids):
	division=[]
	for i in range(len(centroids)):
		Xi=data_set[label==i]
		division.append(Xi)
	return np.array(division)
	pass
def static_visualize(divided_data,centroids):
	color=['cyan','yellow','magenta','green','red','blue','black']
	print(divided_data)
	for i in range(len(divided_data)):
		X=divided_data[i][:,0]
		Y=divided_data[i][:,1]
		color_pnt=i%len(color)
		label='cluster '+str(i)
		# color_cent=(i+1)%len(color)
		plt.plot(X,Y,'o',markersize=4,color=color[color_pnt],label=label)
	X_cen=centroids[:,0]
	Y_cen=centroids[:,1]
	plt.plot(X_cen,Y_cen,'^',color='green',markersize=15)
	plt.legend()
	plt.show()
	pass
def animate_visualize(k,generator,data_set):
	fig,ax=plt.subplots()
	X=data_set[:,0]
	Y=data_set[:,1]
	k=len(centroids)

	color=['cyan','yellow','magenta','green','red','blue','black']
	# data_points=ax.plot(X,Y,'ro')

	plot_cluster=[]
	cluster_0,= plt.plot(X,Y,'o',markersize=4,color=color[0],label='cluster 0')
	plot_cluster.append(cluster_0)

	# plot polygon for each cluster point
	# ???

	from scipy.spatial import Voronoi, voronoi_plot_2d
	
	
	for j in range(1,k):
		color_pnt=j%len(color)
		label='cluster '+str(j)
		cluster_j,=plt.plot([],[],'o',markersize=4,color=color[color_pnt],label=label)
		plot_cluster.append(cluster_j)

	it=0
	plot_centers,= plt.plot([],[],'g^')
	def init():
		return plot_cluster,plot_centers
	def update(generator):
		nonlocal it
		it+=1
		try:
			centroids,clus_label=generator
			clustered_data=divide_data(data_set,clus_label,centroids)
			# fig.clear()
			for i,cluster in enumerate(plot_cluster):
				X=clustered_data[i][:,0]
				Y=clustered_data[i][:,1]
				# color_cent=(i+1)%len(color)
				# plt.plot(X,Y,'o',markersize=4,color=color[color_pnt],label=label)
				# plt.plot(centroids[:,0],centroids[:,1],'g^')
				cluster.set_data(X,Y)
				plot_centers.set_data(centroids[:,0],centroids[:,1])
			return plot_cluster,plot_centers
			pass
		except Exception as e:
			print("Stop")
			return
	animation=FuncAnimation(fig,update,init_func=init,frames=generator,interval=2000)
	plt.show()
	pass
if __name__=='__main__':
	import Kmeans_PP
	centroids=[[1,2],[1,10],[10,2]]
	ndpc=400
	k=3
	data_set=init_data(centroids,ndpc)
	# centroids=init_centroid(data_set,k)
	centroids=Kmeans_PP.init_center(k,data_set)
	generator=animate_kmeans(centroids,data_set,k)
	# print(next(generator))
	animate_visualize(k,generator,data_set)
	# centroids,labels=next(generator)
	# print(labels.shape,centroids.shape)
	# division=divide_data(data_set,labels,centroids)
	# static_visualize(division,centroids)
# Cải tiến giải thuật Kmeans
# Khởi tạo
# Tính toán khoảng cách
# Xác định các khoảng cách với hình tốt hơn
# Loại bỏ nhiễu
# Smoooth Kmean by hierachical
# Kmean combinate with condition that need optimized
# Giải thuật song song cho Kmeans