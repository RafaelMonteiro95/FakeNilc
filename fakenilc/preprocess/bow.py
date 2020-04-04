from sklearn.feature_extraction.text import CountVectorizer
from fakenilc.preprocess import utils
import numpy as np
import pandas as pd

#removes attributes with frequency under min_freq on the dataset
def removeMinFreq(data, labels, min_freq = 1):
	# counting no. of ocurrences per word
	cols_sum = np.sum(data, axis=0)

	#creating an array with indexes of columns that must be deleted 
	del_indexes = []
	#for each val[i] in the cols_sum
	for i, val in zip(range(len(cols_sum)), cols_sum):
		#if that val is smaller than the minimun freq. insert i into the array
		if val < min_freq:
			del_indexes.append(i)

	#deleting columns with minimum frequency smaller than x
	#calls np.delete on the array, asking it to delete all columns with indexes given by del_indexes
	data = np.delete(data,del_indexes,1) 
	labels = np.delete(labels,del_indexes,0)
	return (data, labels)

def removeMinFreqDf(df, min_freq = 1):

	data = np.array(df.values)
	labels = np.array(df.columns.values)

	# counting no. of ocurrences per word
	cols_sum = np.sum(data, axis=0)

	#creating an array with indexes of columns that must be deleted 
	del_labels = []
	#for each val[i] in the cols_sum
	for i, val in zip(range(len(cols_sum)), cols_sum):
		#if that val is smaller than the minimun freq. insert i into the array
		if val < min_freq:
			del_labels.append(labels[i])

	#deleting columns with minimum frequency smaller than x
	#calls np.delete on the array, asking it to delete all columns with indexes given by del_indexes
	return df.drop(del_labels,axis=1)


#normalizes attributes frequencies
def normalizeData(data):
	rows_sum = np.sum(data, axis=1)
	data = (data.T / rows_sum).T
	return data

#used to vectorize a single text with a given vocabulary.
def vectorize(text, vocabulary, preprocessor = None):

	# loading preprocessor
	if preprocessor == None:
		p = utils.preprocessor()

	# Creating bag of words
	vectorizer = CountVectorizer(preprocessor = p.prep, encoding='utf-8', binary = True, vocabulary = vocabulary);

	# getting matrix with words frequencies for each document
	data = np.array(vectorizer.fit_transform([text]).todense());
	labels = np.array(vectorizer.get_feature_names())

	#returns a pandas dataframe with data and labels
	return pd.DataFrame(data,columns = labels)


def loadCount(filenames, min_freq = 1, binary = False, normalize = True):

	# loading preprocessor
	p = utils.preprocessor()

	# Creating bag of words
	vectorizer = CountVectorizer(input = 'filename', preprocessor = p.prep, encoding='utf-8', binary = binary);

	# getting matrix with words frequencies for each document
	data = np.array(vectorizer.fit_transform(filenames).todense());
	labels = np.array(vectorizer.get_feature_names())

	if(min_freq > 1):
		data, labels = removeMinFreq(data, label, min_freq)

	if(normalize):
		data = normalizeData()

	#returns a pandas dataframe with data and labels
	return pd.DataFrame(data,columns = labels)