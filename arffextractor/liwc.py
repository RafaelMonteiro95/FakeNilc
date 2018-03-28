# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
import nltk
import re
import string

class LIWC:

	def __init__(self, LIWCFileName):
		self.LIWCFileName = LIWCFileName

		#dictionary to store classes names and id. Key is ID, name is value
		self.classes = {}
		self.words = {}

		self.start()

	def __repr__(self):
		return str(self.classes)

	def start(self):
		with open(self.LIWCFileName, encoding = 'utf8') as f:
			#reading '%' symbol
			f.readline()

			#reading LIWC classes from file
			for line in f :
				split = line.split()

				#if i've already parsed all classes
				if(split[0] == '%'):
					break; 

				#storing each class by their ID
				self.classes[split[0]] = split[1]

			#reading words classes
			for line in f:
				split = line.split()

				#if i've parsed all words
				if(split[0] == '%'):
					break;

				#creating a new dictionary that stores all classes that a word belongs
				self.words[split[0]] = split[1:]

	def calculateFreqs(self, text, normalized = True, total_normalization = False):
		discarted_words = []
		wordFreqs = {}

		word_list = text.split();

		#initializing all word categories
		for category in self.classes:
			wordFreqs[category] = 0

		#for each word in my text
		for text_word in word_list:

			#if this word exists in our LIWC:
			if text_word in self.words:

				#get all LIWC categories that text_word belongs to
				for category in self.words[text_word]:

					#if we have to normalize with total words
					if(normalized and (not total_normalization)):
						wordFreqs[category] += 1.0/len(word_list)
					else:
						wordFreqs[category] += 1;
			else:
				discarted_words.append(text_word)

		#if we should normalize only with used words
		if(normalized and total_normalization):
			norm_value = len(word_list) - len(discarted_words)

			#normalizes each frequency
			for key in wordFreqs:
				wordFreqs[key] /= norm_value

		return wordFreqs

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
	result = ' '.join([word.lower() for word in result.split() if word not in cachedStopWords])

	return result

#function that loads the corpus and counts LIWC classes frequencies
def loadLiwc(filenames, tags, total_normalization = True):
	result = {'labels': [], 'data': []}

	#loading LIWC
	liwc = LIWC('utils/liwc.txt')

	#preparing result labels
	labels = [liwc.classes[key] for key in liwc.classes]

	result['labels'] = labels

	count = 0
	#processing corpus
	for filename in filenames:
		with open(filename, encoding='utf8') as f:

			#calculates LIWC words frequencies in f, using prep to preprocess the text
			freqs = liwc.calculateFreqs(prep(f.read()),total_normalization = total_normalization)

			freqs_list = [0]*len(labels)

			#each key is a class ID
			for key in freqs:

				classId = key
				className = liwc.classes[key]
				classIndex = labels.index(className)

				freqs_list[classIndex] = freqs[key]

			result['data'].append(freqs_list)

	#inserting labels (TRUE or FALSE) for each instance of the dataset
	for tag,doc in zip(tags,result['data']):
		doc.append(tag)

	result['labels'].append('Trustworthy')

	return result