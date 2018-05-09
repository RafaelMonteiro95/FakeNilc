from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocess import utils
import numpy as np
import pandas as pd


def loadCount(filenames, min_freq = 1, binary = False, normalize = True):

	#loading preprocessor
	p = utils.preprocessor()

	# Creating bag of words
	vectorizer = CountVectorizer(input = 'filename', preprocessor = p.prep, encoding='utf-8', binary = binary);

	# matrix with words frequencies for each document
	data = np.array(vectorizer.fit_transform(filenames).todense());
	labels = np.array(vectorizer.get_feature_names())

	# print('Saída do Vetorizador')
	# print(pd.DataFrame(data[:,:10], columns = labels[:10]) )

	# counting no. of ocurrences per word
	cols_sum = np.sum(data, axis=0)
	# print('Frequência de cada termo')
	# print(pd.DataFrame( [cols_sum[:10]] ,columns = labels[:10]))

	#creating an array with indexes of columns that must be deleted 
	del_indexes = []
	#for each val[i] in the cols_sum
	for i, val in zip(range(len(cols_sum)), cols_sum):
		#if that val is smaller than the minimun freq. insert i into the array
		if val < min_freq:
			del_indexes.append(i)

	# print('Colunas que serão deletadas')
	# print(*[labels[i] for i in del_indexes if i < 10],sep=', ')
	#deleting columns with minimum frequency smaller than x
	#calls np.delete on the array, asking it to delete all columns with indexes given by del_indexes
	data = np.delete(data,del_indexes,1) 
	labels = np.delete(labels,del_indexes,0)

	# print('Após deletar as colunas')
	# print(pd.DataFrame(data[:,:10], columns = labels[:10]) )

	# counting no. of words per document
	rows_sum = np.sum(data, axis=1)
	# print('Número de palavras por documento')
	# print(pd.DataFrame( rows_sum ,columns = ['# de Palavras']))

	if(normalize):
		data = (data.T / rows_sum).T

	# print('Após Normalizar')
	# print(pd.DataFrame(data[:,:10], columns = labels[:10]) )

	rows_sum = np.sum(data, axis=1)
	# print('Soma das linhas')
	# print(pd.DataFrame( rows_sum ,columns = ['Soma das linhas']))

	#returns a dictionary with data and labels
	return pd.DataFrame(data,columns = labels)