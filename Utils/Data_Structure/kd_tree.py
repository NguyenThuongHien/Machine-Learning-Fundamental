import numpy as np
from find_median import med_Of_med
from pprint import pformat
from scipy.spatial.distance import cdist
import math
class Node():
	def __init__(self,value,left,right):
		self.value = value
		self.left = left
		self.right = right
	def __repr__(self):
		# must modify for print tree
		return pformat(tuple([self.value,tuple([self.left,self.right])]))
		pass
	def is_leaf(self):
		if (self.left is None) and (self.right is None):
			return True
		return False
		pass
	
class KDTree():
	def __init__(self,data,root=None):
		self.data = data
		self.root = root
		pass
	def build_tree(self,X,depth=0,leaf_size=10):
		self.leaf_size = leaf_size
		n_samples = X.shape[0]
		n_features = X.shape[1]
		if leaf_size >= n_samples:
			return Node(X,None,None)
		axis = depth%n_features
		X_copy = np.copy(X)
		# partition
		idx_med = n_samples//2
		partition = X_copy[np.argpartition(X_copy[:,axis],idx_med)]
		X_left = partition[0:idx_med]
		X_right = partition[idx_med+1:n_samples]
		# contruct child
		left_child = self.build_tree(X_left,depth+1,leaf_size)
		right_child = self.build_tree(X_right,depth+1,leaf_size)
		self.root = Node(partition[idx_med],left_child,right_child)
		return self.root
		pass
	def query(self,root,target,axis=0,hyperplane=None,k_nearest=1,current_best=[],best_dist=[],
									is_root=True,return_dist=True):
		# Algorithm
		'''
		Traverse KDtree
		+From the root,go down the tree
			+If reach leaf : search bruteforce
			  +At point visiting, consider it to result array
			+Else:
			  +Recursive to traverse tree
			  +At point visiting, consider it to result array
			  +After receive result from recursive call:
			  	+Compute distance to splitting hyperplane and consider to removing search in other side
				+If i points have current best > distance: search k-i nearest on other side
				+Else, prune other branch
				+Return
		'''
		if root.is_leaf():
			# brute force
			leaf = root.value
			dis_leaf = cdist(target,leaf)
			# best_dist = cdist(target,current_best)
			insert_idx = np.searchsorted(best_dist,dis_leaf)
			merge_arr = np.insert(current_best,insert_idx,leaf)
			results = merge_arr[k_nearest]
			if return_dist == True:
				merge_dis = bp.insert(best_dist,insert_idx,dis_leaf)
				merge_dis = merge_dis[k_nearest]
			return results,merge_dis
		else:
			distance = np.linalg.norm(target.value-root.value)
			if is_root == True:
				current_best = np.array([root.value])
				best_dist = np.array([distance])
			else:
				position = np.searchsorted(best_dist,distance)
				insert_node = np.insert(current_best,position,root.value)
				insert_dist = np.insert(best_dist,position,distance)
				if len(insert_dist) < k_nearest:
					current_best = insert_node
					best_dist = insert_dist
				else:
					current_best = insert_node[0:k_nearest]
					best_dist = insert_dist[0:k_nearest]
			n = len(self.data[0])
			axis = axis % n
			hyperplane = root.value[axis]
			# can create a func for brevity
			if target.value[axis] < root.value[axis]:
				current_best,best_dist = self.query(root.left,target,axis+1,hyperplane,k_nearest,current_best,best_dist)
				if hyp is None:
					return current_best,best_dist
				else:
					dist_to_hyp = np.abs(current_best-hyperplane)
					diff_dist = best_dist - dist_to_hyp
					if np.all(diff_dist < 0):
						return current_best,best_dist
					else:
						mask = diff_dist > 0
						fault_best = current_best[mask]
						fault_dist = best_dist[mask]
						kn = len(fault_best)
						hyp = None
						sub_best,sub_dist = self.query(root.right,target,axis+1,hyp,kn,fault_best,fault_dist)
						current_best,best_dist = KDTree.merge_arr(current_best,sub_best,best_dist,sub_dist,k_nearest)
						return current_best,best_dist
			else:
				current_best,best_dist = self.query(root.right,target,axis+1,hyperplane,k_nearest,current_best,best_dist)
				if hyp is None:
					return current_best,best_dist
				else:
					dist_to_hyp = np.abs(current_best-hyperplane)
					diff_dist = best_dist - dist_to_hyp
					if np.all(diff_dist < 0):
						return current_best,best_dist
					else:
						mask = diff_dist > 0
						fault_best = current_best[mask]
						fault_dist = best_dist[mask]
						kn = len(fault_best)
						hyp = None
						sub_best,sub_dist = self.query(root.right,target,axis+1,hyp,kn,fault_best,fault_dist)
						current_best,best_dist = KDTree.merge_arr(current_best,sub_best,best_dist,sub_dist,k_nearest)
						return current_best,best_dist
			return current_best,best_dist
		pass
	def query_range(self,root,range):
		'''
		how to denote a range? an inequality
		Algorithm:
		query_range(root,range):
			if root is leaf node
				if root in range return root
			else
				if leafside completely IN_RANGE(range) return ALLNODE(left side)
				if rightside completely IN_RANGE(range) return ALLNODE(rightside)
				if leafside INTERSECT(range): 
					left_result = QUERY_RANGE(root.left,range)
				if rightside INTERSECT(range):
					right_result = QUERY_RANGE(root.right,range)
				return UNION(left_result,right_result)
		'''
		pass
	def insert(self,balance=True):
		pass
	def add(self,value,depth=0,balance=False):
		'''
		Algorithm
		+Traverse the tree until reach to leaf node
		+Add to the tree:
			+At leaf node:
				_If len(leaf)+value > self.leaf_size: reconstruct node based on leaf
		'''
		if self.root.is_leaf():
			old_node = self.root
			leaf = self.root.value
			if len(leaf) + 1 > self.leaf_size:
				concatenation = np.append(leaf,value,axis=0)
				mid_pos = len(concatenation)//2
				partition = np.partition(concatenation,mid_pos)
				self.root = Node(partition[mid_pos],partition[:mid_pos],partition[mid_pos+1:])
			else:
				self.root = np.append(leaf,value,axis=0)
			del old_node
		else:
			if value < self.root.value:
				self.left.add(value,depth+1)
			else:
				self.right.add(value,depth+1)
		pass
	def remove(self,root=self.root,value=None,depth=0):
		if root.is_leaf():
			return(root.value[np.all(np.not_equal(root.value,value),axis=1)])
		if KDTree.is_equal(root.value,value):
			if root is None:
				return None
			elif not(root.right is None):
				min_value = self.find_min(root.right,depth+1,depth+1)
				root.value = min_value
				root.right = self.remove(root.right,min_value,depth+1)
			else:
				min_value = self.self.find_min(root.left,depth+1,depth+1)
				root.value = min_value
				root.right = self.remove(root.left,min_value,depth+1)
		else:
			axis = depth % len(root.value)
			if value[axis] < root.value[axis]:
				root = self.remove(root.left,value,depth+1)
			else:
				root = self.remove(root.right,value,depth+1)
		return root
		pass
	def find_min(self,root,axe=0,depth=0):
		if root is None:
			return None
		if root.is_leaf():
			value = root.value
			axis = depth % len(value[0])
			idx = np.argmin(root.value[:,axis])
			return value[idx]
		axis = depth % len(root.value)
		if axis == axe:
			if root.left is None:
				return root.value
			return self.find_min(root.left,axe,depth+1)
		else:
			return KDTree.min_along(root.value,self.find_min(root.left,axe,depth+1),
				self.find_min(root.right,axe,depth+1),axe)
		pass
	def get_data(self,root=self.root):
		if root.is_leaf():
			return root.value
		else:
			left = self.get_data(root.left)
			right = self.get_data(root.right)
			return np.append(np.append(left,[root.value],axis=0),right,axis=0)
		pass
	@staticmethod
	def find_replacement(node):
		'''
		+if self in leaf return None
			+if len(leaf) is 1, assign leaf is None, free node
		+elif node have right child: 
			*find minimum of tree rigthchild
			*copy node
			*assign copy node to node
			*recur to delete minimum node in right child of current node
		+else
			*find min node in left child
			*assign copy to node
			*recur to delete in left child, and assign right to left child deleted
		'''
		pass
	def __repr__(self):
		return pformat(self.root)
		pass
	@staticmethod
	def merge_arr(A1,A2,B1,B2,k):
		idx = np.searchsorted(B1,B2)
		B = np.insert(B1,idx,B2)
		A = np.insert(A1,idx,A2)
		return A[0:k],B[0:k]
		pass
	@staticmethod
	def brute_force(A,target,current_best,current_dist,k):
		dist_A = cdist(target,A)
		return KDTree.merge_arr(A,current_best,current_dist,dist_A,k)
		pass
	@staticmethod
	def copy_node(node1,node2):
		node1.value = np.copy(node2)
		pass
	@staticmethod
	def is_equal(value1,value2):
		return np.array_equal(value1,value2)
		pass
	@staticmethod
	def min_among(value1,value2,value3,axis=0):
		result = value1
		if not(value2 is None) and (value2[axis] <= result[axis]):
			result = value2
		if not(value3 is None) and (value2[axis] <= result[axis]):
			result = value3
		return result
		pass
	@staticmethod
	# check completely in range
	def in_range(range1,range2):
		return(np.all(np.less(range1,range2)),axis=0) == np.array([True,False])
		pass
	def union(arr1,arr2):
		return np.append(arr1,arr2,axis=0)
		pass
	def have_intersect(range1,range2):
		combination = np.array(range1,range2)
		return np.all(np.max(combination[:,:,1],axis=0) - np.min(combination[:,:,0],axis=0)>=0)
		pass
if __name__ == '__main__':
	A = np.array([[2,3,6],[0,4,5],[6,3,7],[2,5,1],[4,1,2],[5,1,6]])
	kdtree = KDTree(A)
	kdtree.build_tree(A,leaf_size=2)
	print(kdtree)