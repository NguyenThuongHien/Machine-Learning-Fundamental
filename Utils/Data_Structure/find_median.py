def nlogn_median(list_):
	list_tmp = sorted(list_)
	return list_tmp[(len(list_tmp)//2)]
	pass

# inspired from philosophy quicksort
# average complexity is O(n)
'''
	Parameters:
	  list_: list needed choose median
	  pivot_fnc: choice function
	Utils function:
	  quick_select: function get k smallest element in list 
	  				or element at k position in sorted list
	  rotate_list: function that tranform list satisfied:
	  				left side is smaller, otherwise
'''
import random
def quick_med(list_,pivot_fnc=random.choice):
	mid_idx = len(list_)//2
	return quick_select(list_,mid_idx,pivot_fnc)
	pass
def quick_select(list_,k,pivot_fnc):
	if len(list_) == 1:
		assert k == 0
		return list_[0]
	else:
		n = len(list_)
		id_pivot = pivot_fnc(range(n))
		list_,idx_pivot,pivot = rotate_list(list_,id_pivot)
		if idx_pivot == k: return list_[k]
		elif idx_pivot < k: return quick_select(list_[idx_pivot+1:n],k-idx_pivot-1,pivot_fnc)
		elif idx_pivot > k: return quick_select(list_[0:idx_pivot],k,pivot_fnc)
	pass
def rotate_list(list_,idx_pivot):
	n = len(list_)
	# swap
	pivot = list_[idx_pivot]
	list_[idx_pivot],list_[n-1] = list_[n-1],list_[idx_pivot]

	i = 0
	j = n-2
	while i<=j:
		while (i<n-1) and (list_[i] < pivot): i+=1
		while (j>=0) and (list_[j] >= pivot): j-=1
		# swap
		if i<j:
			list_[i],list_[j] = list_[j],list_[i]
			i+=1
			j-=1
	# swap
	list_[i],list_[n-1] = list_[n-1],list_[i]
	return list_,i,pivot
	pass
'''
	Parameters:
	  a: list contain elements
	  k: getting position
'''
# Complexity in worst case O(n)
def med_Of_med(A,k):
	n=len(A)
	if n<=5:
		return sorted(A)[k]
	# split A to chunk with chunksize 5
	chunks = [A[j:j+5] for j in range(0,n,5) if j+5<n]
	sorted_chunk = [sorted(chunk) for chunk in chunks]
	medians = [chunk[2] for chunk in sorted_chunk]
	# find median of medians for pivot
	pivot = med_Of_med(medians,len(medians)//2)
	idx_pv = partition(A,pivot)
	if idx_pv == k:
		return pivot
	elif idx_pv < k:
		return med_Of_med(A[idx_pv+1:n],k-idx_pv-1)
	elif idx_pv >k: return med_Of_med(A[0:idx_pv],k)
	pass
def partition(A,pivot):
	# barrier : check for loop reached to first of position?
 	barrier = 0
 	# mark:idicate position partitioned
 	mark = 0
 	for i in range(len(A)-1):
 		if (barrier == 0) and (A[i] == pivot):
 			A[i],A[-1] = A[-1],A[i]
 			barrier = 1
 		if A[i] < pivot:
 			A[i],A[mark] = A[mark],A[i]
 			mark+=1
 	A[-1],A[mark] = A[mark],A[-1]
 	return mark
 	pass

def get_idx(A,value):
	return A.index(value)
	pass
def swap(A,i,j):
	A[i],A[j] = A[j],A[i]
	pass
if __name__ == '__main__':
	import sys
	# sys.setrecursionlimit(10000)
	a = [2, 0, 6, 2, 4, 5]
	i= med_Of_med(a,len(a)//2)
	print("len A:",len(a)//2)
	print("med: ",i)
	print("med: ",nlogn_median(a))
	print(sorted(a))