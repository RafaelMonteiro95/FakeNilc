from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from extractor import preprocessing
import pandas as pd

def loadCount(filenames, tags, max_features = None, normalize = False, total_normalization = True):

	# Creating bag of words
	vectorizer = CountVectorizer(input = 'filename', preprocessor = preprocessing.prep, encoding='utf-8', max_features=max_features);

	# matrix with words frequencies for each document
	data = vectorizer.fit_transform(filenames).todense();
	labels = vectorizer.get_feature_names()

	#returns a dictionary with data and labels
	return pd.DataFrame(data,columns = labels)

def loadTfidf(filenames, tags, max_features = None, normalize = False, total_normalization = True):

	# Creating bag of words
	vectorizer = TfidfVectorizer(input = 'filename', preprocessor = preprocessing.prep, encoding='utf-8',max_features=max_features);

	# matrix with words frequencies for each document
	data = vectorizer.fit_transform(filenames).todense();
	labels = vectorizer.get_feature_names()

	#returns a dictionary with data and labels
	return pd.DataFrame(data,columns = labels)