# import numpy as np
# # f= open('mysave1.npy','wb')
# # z=np.array([4,5])
# # a=np.array([1,2])
# # b=np.array([2,3])
# # np.save(f,a)
# # np.save(f,b)
# # f.close()
# # np.save('mysave.npy',a)
# # np.save('mysave.npy',b)

# # while 1:
# # 	try:
# # 		a=np.load('mysave.npy')
# # 	except  as e:
# # 		raise e
# # 	pass
# # Unsupervised Pose Flow Learning for Pose Guided Synthesis

# # print(np.stack(np.load(f) for _ in range(10)))
# # f = open('mysave1.npy','rb')
# # while 1:
# # 	try:
# # 	 	print(np.load(f))
# # 	except Exception as e:
# # 	 	print('stoped')
# # 	 	break

# # for i in range(5):
# # 	new_d=np.load('mysave.npy')
# # 	print(new_d)
# # 	print(d)
# # 	if np.array_equal(d,new_d): break
# # 	d = new_d
# # print(i)
# # print(d)

# # a = np.array([[1,5],[0,1]])
# # b = np.array([[1,2,3],[2,3,4]])
# # a = np.array([2,3,5,6,7])
# # print(np.sum(a))
# def edit_distance(s1,s2):
# 	pass
# def min_dis():
# 	pass
# a,b=np.ogrid[0:7:3,0:5:3]
# # print(a,b)
# x1, y1 = np.meshgrid(np.arange(1, 11, 2), np.arange(-12, -3, 3),indexing='ij')
# # print(x1,'\n',y1)
# a = np.tile(np.arange(3),(2,1))
# b = np.array([1, 1, 2, 2, 3, 3])
# # b[:,[2,3]]=[5,6]
# # print(b)
# # print(np.insert(a,[1,1],[[3,3],[2,4]],axis=1))
# # print(np.insert(a,[1,1],[3,2],axis=1))
# # print(a)
# # b[:2]=6
# # print(b)
# # print(c)
# # # a[:2]=[6]
# # print(a)

# a=np.array([[1, 1],
#        [2, 2],
#        [3, 3]])
# print(a.shape)
# b=np.insert(a, 1, [1,2,3], axis=1)
# # print(b)

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6,9],[7,8,9]])
# # print(np.concatenate((a,b),axis=1))

# # print(__name__,__package__)
# import sys
# # print(sys.modules)

# a = np.array([1,2,3])
# b=np.array([1,2,4])
# # print(a*b)
# from scipy.spatial.distance import cdist
# a = np.array([[3,4],[2,7],[2,5]])
# b = np.array([[1,2],[2,3]])
# c = cdist(a,b)


# # print(c)
# c=np.argsort(c,axis=0)[:2,:]
# print(c)
# d = np.tile(np.arange(c.shape[1]),(c.shape[0],1))
# k = (c,d)
# # print(a[k])

# a =np.array([5,3,2,9,1,0])
# m = np.array([[0,1],[2,4]])
# # print(np.take(a,m.T))

# values=np.array([1,2,3,4,5,6,7,8,9,10])
# valcats=np.array([101,301,201,201,102,302,302,202,102,301])
# categories=np.array([101,102,201,202,301,302])
# # print(np.searchsorted(categories, valcats))
# # print(np.bincount(np.searchsorted(categories, valcats), values))

# A = np.array([[1,6],[1,3],[3,3]])

# m = A.shape[1]    
# n = A.max()+1
# A1 = A + (n*np.arange(m))
# # out = np.bincount(A1.ravel(),minlength=n*m).reshape(m,-1).T

# # print("A1:\n",A1.ravel('F'))
# # category = np.unique(A1.ravel('F'))
# # print(category)
# # normalize_bin = np.searchsorted(category,A1.ravel('F'))
# # print("nom_bin:\n",normalize_bin)
# # n = normalize_bin.max()+1
# w = np.array([[1.2,1.4],[2.1,3.1],[6.0,3.5]])
# weight = w.flatten('F')
# out = np.bincount(A1.ravel('F'),weight,minlength=m*n)
# out_ = out.reshape(m,-1).T
# print("out \n",out)
# print("out flat\n",out_)
# print(A1)
# print(A1.flatten('F'))

# a = np.array([2,5,7,9,8])
# idx = [[1,2],[0,3]]
# # print(np.take(a,idx))
# # print(a[None,:])
# a = [[1,0,0],[2,0,0],[3,1,0]]
# # print(np.unique(a,axis=1))

# # how to flat weight with 0 coefficient?
# rng = np.random.RandomState(0)
# X = rng.random_sample((10, 3))
# print(X)
# print(X[:1])
# import numpy as np
# a = np.array([[1,2],[2,4],[4,5]])
# b= np.copy(a,'K')
# a[0] = [2,3]
# print(a)
# print(b)
# a = a.ravel()
# print(a)
# print(list(a).index(4))
import numpy as np
x = np.array([[0,1,3, 4, 2, 1,4],[2,4,5,1,4,2,0]])
# print(x[:,np.argpartition(x[0], 3)])
X = np.array([[5,3,4],[2,5,1]])
idx = np.argpartition(X[0],1)
# print(idx)
X_par = X[:,idx]
# print(X_par)
# insert np
a = np.array([1,4,4,6])
b = [2,5,7]
idx = np.searchsorted(a,b)
# print(idx)
a_ = np.array([[1,1],[4,4],[4,5],[6,6]])
b = np.array([[2,2],[5,5],[7,7]])
# print(np.insert(a_,idx,b,axis=0))
from scipy.spatial.distance import cdist
a = np.array([1,2,3,2,3,4])
b = np.array([3,3,4])
# print(np.sqrt((a-b)*(a-b)))
# a = np.array([[1,2]])
# b=np.append(a,[[0,1]],axis=0)
# print(np.searchsorted(a,5))
# print(np.all(sub>0))
# print(a[:2])
a=np.array([[1,2],[4,3]])
b=np.array([[1,2],[2,3],[3,1]])
print(cdist(a,b))
# print(a[np.all(np.not_equal(a,b),axis=1)])
# c=np.array([a,b])
# print(1/2+1)
# print(np.random.rand(10))
# X=np.array([[ 0.48252164,0.12013048],
#  [ 0.77254355,0.74382174],
#  [ 0.45174186,0.8782033 ],
#  [ 0.75623083,0.71763107],
#  [ 0.26809253,0.75144034],
#  [ 0.23442518,0.39031414]])
# print(np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X))
# print(X[np.random.permutation(X.shape[0])])
# print(np.random.permutation(X.shape[0]))
# print(np.random.permutation(X.shape[0]))
# print(np.random.permutation(X.shape[0]))
b=np.zeros(2)
print(np.min(a,axis=0,keepdims=True))
# np.min(a,axis=1,out=b)
print(b)