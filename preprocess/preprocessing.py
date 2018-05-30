from nltk.corpus import stopwords
import unicodedata
import nltk
import re
import string

def prep(s, useStopWords = True, stem = True):

	#preparation
	stemmer = nltk.stem.SnowballStemmer('portuguese')
	translator = str.maketrans({key:' ' for key in string.punctuation});
	
	#removing ponctuation
	result = s.translate(translator);

	#removing numbers
	result = re.sub('[0-9]', '' , result)

	#removing stopwords
	if useStopWords and stem:
		result = ' '.join([stemmer.stem(word.lower()) for word in result.split() if word not in stopwords.words('portuguese')])
	elif useStopWords:
		result = ' '.join([word.lower() for word in result.split() if word not in stopwords.words('portuguese')])
	elif stem:
		result = ' '.join([stemmer.stem(word.lower()) for word in result.split() if word not in stopwords.words('portuguese')])
	else:
		result = ' '.join([word.lower() for word in result.split()])
	return result