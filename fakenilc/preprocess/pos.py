# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#supress some warnings about type conversion
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from nlpnet import POSTagger


def countTags(text, tagger, normalize=True):

	wordcount = 0

	#pos tags used by nlpnet
	pos_tags = {'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0}

	#counting frequencies
	#for each resulting tuple from the tagging method
	tagged_text = tagger.tag(text)
	# print(tagged_text)
	for res in tagged_text:

		#counting number of tagged words in text
		wordcount += len(res)

		for word_result in res:
			#sometimes one word gets more than one tag. Splitting it into two or more tags
			split_result = word_result[1].replace('+',' ').split()

			#increase the frequency of each tag
			for tag in split_result:
				pos_tags[tag] += 1

	#saving the tags count to a Numpy array
	# result = np.array(pos_tags)
	result = np.array(list(pos_tags.values()))
	# print(wordcount)
	if(normalize):
		try:
			result = result / wordcount
		except RuntimeWarning:
			import ipdb; ipdb.set_trace()

	# for tag,value in zip(pos_tags,result):
	# 	print("'{0}': {1:.2}".format(tag,value), end = ', ')
	# print()

	return result

def vectorize(text, tagger = None):

	if tagger == None:
		tagger = POSTagger(r'var/nlpnet', language='pt')

	labels = list({'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0}.keys())
	freqs = countTags(text,tagger)

	return pd.DataFrame([freqs],columns=labels)


#function that loads the corpus and counts LIWC classes frequencies
def loadPos(filenames):

	data = []

	#loading nlpnet	
	tagger = POSTagger(r'fakenilc/var/nlpnet', language='pt')

	labels = list({'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0}.keys())

	#loading files
	for filename in filenames:
		with open(filename, encoding='utf8') as f:
			#preprocesses the text read in f using prep()
			#then counts the frequencies using the tagger
			#returns a list with frequencies
			freqs = countTags(f.read(),tagger)
			#then appends this list into the data segment of the result dict
			data.append(freqs)

	return pd.DataFrame(data,columns=labels)