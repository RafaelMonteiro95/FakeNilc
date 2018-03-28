from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import unicodedata
import nltk
import re
import string

def prep(s):

	#preparation
	cachedStopWords = stopwords.words('portuguese')
	stemmer = nltk.stem.SnowballStemmer('portuguese')
	translator = str.maketrans({key:None for key in string.punctuation});
	
	#removing ponctuation
	result = s.translate(translator);
	#removing numbers
	result = re.sub('[0-9]', '' , s)
	#removing stopwords
	result = ' '.join([stemmer.stem(word.lower()) for word in result.split() if word not in cachedStopWords])

	return result


def loadCount(filenames, tags, max_features = None):

	# Creating bag of words
	vectorizer = CountVectorizer(input = 'filename', preprocessor = prep, encoding='utf-8', max_features=max_features);

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

def loadTfidf(filenames, tags, max_features = None):

	# Creating bag of words
	vectorizer = TfidfVectorizer(input = 'filename', preprocessor = prep, encoding='utf-8',max_features=max_features);

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