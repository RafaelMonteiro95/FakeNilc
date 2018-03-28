from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from arffextractor import preprocessing

def loadCount(filenames, tags, max_features = None, normalize = False, total_normalization = True):

	# Creating bag of words
	vectorizer = CountVectorizer(input = 'filename', preprocessor = preprocessing.prep, encoding='utf-8', max_features=max_features);

	# matrix with words frequencies for each document
	frequencies = vectorizer.fit_transform(filenames).todense().tolist();
	words = vectorizer.get_feature_names()

	#inserting labels (TRUE or FALSE) for each instance of the dataset
	for tag,doc in zip(tags,frequencies):
		doc.append(tag)

	#inserting Trustworthy as a attribute of the instances
	words.append('Trustworthy')

	#returns a dictionary with data and labels
	return {'labels':words, 'data':frequencies}

def loadTfidf(filenames, tags, max_features = None, normalize = False, total_normalization = True):

	# Creating bag of words
	vectorizer = TfidfVectorizer(input = 'filename', preprocessor = preprocessing.prep, encoding='utf-8',max_features=max_features);

	# matrix with words frequencies for each document
	frequencies = vectorizer.fit_transform(filenames).todense().tolist();
	words = vectorizer.get_feature_names()

	#inserting labels (TRUE or FALSE) for each instance of the dataset
	for tag,doc in zip(tags,frequencies):
		doc.append(tag)

	#inserting Trustworthy as a attribute of the instances
	words.append('Trustworthy')

	#returns a dictionary with data and labels
	return {'labels':words, 'data':frequencies}