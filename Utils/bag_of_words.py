'''
	Parameters:
	----------
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
'''
import os
import numpy as np
import pickle
from collections import OrderedDict

class BOW(object):
	def __init__(doc_path,vocab_path,vector_path,tokenizer=None):
		self.__doc_path = doc_path
		self.__vocab_path = vocab_path
		self.__vector_path = vector_path
		self.__initiated = False
		self.__tokenizer = tokenizer
		pass
	def get_parameters(self):
		return self.__doc_path,self.vocab_path,self.__vector_path,self.has_initiated
		pass
	def set_parameters(self,doc_path='',vocab_path='',vector_path=''
										,initiated=None,tokenizer=None):
		if doc_path != '':
			self.__doc_path = doc_path
		if vocab_path != '':
			self.__vocab_path = vocab_path 
		if vector_path != '':
			self.__vector_path = vector_path
		if not(initiated is None):
			self.__initiated = initiated 
		if not(initiated is None):
			self.__tokenizer = tokenizer
		pass
	def __get_file_set(self):
		file_set = os.list_dir(self.__doc_path)
		return file_set
		pass
	def __get_number_doc(self):
			return len(self.__get_file_set())
		pass
	def __tokenize(self,file_path):
		tokenizer = self.__tokenizer
		word_list = tokenizer.tokenize(file_path)
		return word_list
		pass
	def __get_vocabulary(self,file_set):
		vocabulary=np.array([])
		for file in file_set:
			file_path = os.path(self.__doc_path,file)
			word_set = np.unique(self.__tokenize(file_path))
			vocabulary = np.union1d(vocabulary,word_set)
		vocab_dict = {(k,v) for v,k in enumerate(vocabulary)}
		return vocabulary,vocab_dict
		pass
	def __save_vocabulary(self,path,vocab_dict):
		pickle.dump(vocab_dict,open(path,'wb'))
		pass
	def __compute_BOW(self,file_path,vocab_dict):
		doc2vec = np.zeros(len(vocab_dict))
		word_list = self.__tokenize(file_path)
		for word in word_list:
			doc2vec[vocab_dict[word]] += 1
		return doc2vec
		pass
	def __save_BOW(self):
		# Can use hdf5: https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr
		file_set = self.__get_file_set()
		vocabulary,vocab_dict = self.__build_vocabulary(file_set)
		file_obj = open(self.__vector_path,'ab')
		for file in file_set:
			file_path = os.path(self.__doc_path,file)
			doc2vec = self.__compute_BOW(file_path,vocab_dict)
			np.save(file_obj,doc2vec)
		file_obj.close()
		pass
	def load_BOW(self):
		doc2vecs = []
		file_obj = open(self.__vector_path,'rb')
		while 1:
			try:
				doc2vecs.append(np.load(file_obj))
			except Exception as e:
				break
		return doc2vecs
		pass
	def load_vocab_dict(self):
		pickle.load(open(self.__vocab_path,'rb'))
		pass
	def get_BOW(self,doc_path):
		vocab_dict = self.load_vocab_dict()
		bow_embedding = np.zeros(len(vocab_dict))
		word_list = self.__tokenizer.tokenize(doc_path)
		for word in word_list:
			if word in vocab_dict:
				idex = vocab_dict[word]
				bow_embedding += 1
		return bow_embedding
		pass
	def train(self,vector_path=None):
		if vector_path is None:
			vector_path = self.__vector_path
		file_set = self.__get_file_set()
		vocabulary,vocab_dict = self.__build_vocabulary(file_set)
		self.__save_BOW(vector_path)
		pass
#Have not yet debug