# Color Quantization with Kmeans
#Step through:
	 # load img
	 # get matrix pixel of img
	 # reshape to (w*h,3) get feature for a pixel in 3D point RGB
	 # Run Kmeans: get co-ordinate center, and labels of point in img
	 # Replace value of each pixel by value of respective center
	 # Reshape to original shape of img
from sklearn.cluster import KMeans
from matplotlib.pyplot import imread
import numpy as np
import matplotlib.pyplot as plt

img=imread('girl.jpg')
w=img.shape[0]
h=img.shape[1]
channel=img.shape[2]
X=img.reshape(w*h,channel)

n_cluster=7
kmean=KMeans(n_cluster).fit(X)
labels=kmean.labels_
centers=kmean.cluster_centers_
X_0=np.zeros_like(X)
for k in range(n_cluster):
	X_0[np.where(labels==k)]=centers[k]
# X_0[np.where(labels==1)]=X[np.where(labels==1)]
# X_0[np.where(labels!=1)]=[255,255,255]
img_reduced=X_0.reshape(w,h,channel)
plt.imshow(img_reduced)
plt.axis('off')
plt.show()