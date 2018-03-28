# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
import nltk
import re
import string

#supress some warnings about type conversion
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from nlpnet import POSTagger


def prep(s, useStopWords = False):

	#preparation
	cachedStopWords = stopwords.words('portuguese')
	stemmer = nltk.stem.SnowballStemmer('portuguese')
	translator = str.maketrans({key:' ' for key in string.punctuation});
	
	#removing ponctuation
	result = s.translate(translator);

	#removing numbers
	result = re.sub('[0-9]', '' , result)

	#removing stopwords
	if useStopWords:
		result = ' '.join([word.lower() for word in result.split() if word not in cachedStopWords])
	else:
		result = ' '.join([word.lower() for word in result.split()])

	return result


def countTags(text, tagger, normalize=False):

	wordcount = 0

	#pos tags used by nlpnet
	pos_tags = {'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0}

	#counting frequencies
	#for each resulting tuple from the tagging method
	for res in tagger.tag(text):
		for word_result in res:
			wordcount += 1
			#sometimes one word gets more than one tag. Splitting it into two or more tags
			split_result = word_result[1].replace('+',' ').split()

			#increase the frequency of each tag
			for tag in split_result:
				pos_tags[tag] += 1

	result = list(pos_tags.values())

	for i in range(len(result)):
		result[i] /= wordcount

	return result


#function that loads the corpus and counts LIWC classes frequencies
def loadPos(filenames, tags, normalize=False):

	result = {'labels': [], 'data': []}

	#loading nlpnet	
	tagger = POSTagger(r'utils\nlpnet', language='pt')

	result['labels'] = list({'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0}.keys())

	result['labels'].append('Trustworthy')


	# print(result['labels'])
	#loading files
	for filename, tag in zip(filenames,tags):
		with open(filename, encoding='utf8') as f:
			#preprocesses the text read in f using prep()
			#then counts the frequencies using the tagger
			#returns a list with frequencies
			freqs = countTags(prep(f.read(),useStopWords = False),tagger, normalize=False)
			#then appends the TrustWorthy tag in this list
			freqs.append(tag)
			#then appends this list into the data segment of the result dict
			result['data'].append(freqs)

	return result