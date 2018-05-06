from nltk.corpus import stopwords
import unicodedata
import nltk
import re
import string

def prep(s, useStopWords = True, stem = True):

	#preparation
	cachedStopWords = stopwords.words('portuguese')
	stemmer = nltk.stem.SnowballStemmer('portuguese')
	translator = str.maketrans({key:' ' for key in string.punctuation});
	
	#removing ponctuation
	result = s.translate(translator);
	# print('After removing punctuation:')
	# print(result)

	#removing numbers
	result = re.sub('[0-9]', '' , result)
	# print('After removing numbers:')
	# print(result)

	#removing stopwords
	if useStopWords and stem:
		# result = result.lower()
		# print('After converting to lower case')
		# print(result)
		# result = ' '.join([word for word in result.split() if word not in cachedStopWords])
		# print('After removing stopwords')
		# print(result)
		# result = ' '.join([stemmer.stem(word) for word in result.split()])
		# print('After stemming stopwords')
		# print(result)
		result = ' '.join([stemmer.stem(word.lower()) for word in result.split() if word not in cachedStopWords])
	elif useStopWords:
		result = ' '.join([word.lower() for word in result.split() if word not in cachedStopWords])
	elif stem:
		result = ' '.join([stemmer.stem(word.lower()) for word in result.split() if word not in cachedStopWords])
	else:
		result = ' '.join([word.lower() for word in result.split()])
	return result