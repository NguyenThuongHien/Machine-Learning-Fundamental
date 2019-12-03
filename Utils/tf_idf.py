# tf idf
"""
	Parameters init:
	---------------
	doc_path: string
		path to directory contain all document set
	vocab_path: string
		path where save vocabulary of document set
	vector_path: string
		path where save each vectorized document
	initiated: boolean
		check instance of BOW has trained?
	tokenizer: Tokenizer
		tokenize each document	

"""
# Can use BOW class and numpy to compute tf-idf
# drawbacks of this method are consumed memory by sparse matrix
# but leverage power computing of numpy

# Improve: Use sparse array

from bag_of_words import BOW
import numpy as np

class TF_IDF(BOW):
	def __init__(doc_path,vocab_path,tfidf_path,tf_path,idf_path,tokenizer=None):
		super(BOW,self).__init__(doc_path,vocab_path,tfidf_path,tokenizer)
		self.__tf_path = tf_path
		self.__idf_path = idf_path
		pass
	def __get_vocabulary(self,file_set):
		vocabulary = np.array([])
		docset_freq = []
		for file in file_set:
			file_path = os.path(self.__doc_path,file)
			word_list = self.__tokenize(file_path)
			word_set,tf_doc = np.unique(word_list,return_counts=True)
			freq_dict = {(word,freq) for word,freq in zip(word_set,tf_doc)}
			len_doc = len(word_list)
			docset_freq.append((freq_dict,len_dict))
			vocabulary = np.union1d(vocabulary,word_set)
		vocab_dict = {(k,v) for v,k in enumerate(vocabulary)}
		return vocabulary,vocab_dict,docset_freq
		pass
	def __compute_tf(self,docset_freq):
		for freq_dict,lendoc in docset_freq:
			for k in freq_dict.keys():
				freq_dict[k] /= lendoc
		tf = docset_freq
		return tf
		pass
	def __compute_idf(self,tf):
		idf = {}
		for freqdoc in tf:
			for word in freqdoc.keys():
				if not(word in idf):
					idf[word] = 1
				else:
					idf[word] += 1
		return idf
		pass
	def __compute_tfidf(self,tf,idf):
		for freq_doc in tf:
			for word in tf.keys():
				tf[word]*idf[word]
		return tf
		pass
	def __save_tf(self,tf):
		np.save(self.__tf_path,tf)
		pass
	def __save_idf(self,idf):
		pickle.dump(idf,open(self.__idf_path,'rb'))
		pass
	def __save_tfidf(self,tf_idf):
		np.save(self.__vector_path,tf_idf)
		pass
	def get_tfidf(self,idf,doc_path):
		bow = self.__get_BOW(doc_path)
		vocab_dict = self.load_vocab_dict()
		tf = bow/np.sum(bow)
		idf_ = self.get_idf()
		idf = np.zeros(len(vocab_dict))
		for word in idf_:
			idx = vocab_dict[word]
			idf[idx] = idf_[word]
		tf_idf = tf*idf
		pass
	def get_idf(self):
		return(pickle.load(open(self.__idf_path,'rb')))
		pass
	def train(self,tf_path):
		file_set = self.__get_file_set()
		vocabulary,vocab_dict,docset_freq = self.__get_vocabulary()
		tf = self.__compute_tf(docset_freq)
		idf = self.__compute_idf(tf)
		tf_idf = self.__compute_tfidf(tf,idf)
		self.__save_vocabulary(vocab_dict)
		self.__save_tf(tf)
		self.__save_idf(idf)
		self.__save_tfidf(idf)
		pass
# Problems: 
# 	How to save embedding document effectively?
# 		1.np.save
# 		2.pickle
# 		3.hdf5
# 	Compute tf-idf parallel?
# 	1.np
# 	2.multiprocessing in python