# -*- coding: utf-8 -*-

from extractor import preprocessing
import pandas as pd
import numpy as np
import string
import re

#supress some warnings about type conversion
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from nlpnet import POSTagger

#
#pd.concat([df1, df2.drop('id',axis=1)],axis=1,join='inner').set_index('id')


def loadMetricsCSV(csv_filenames):

	#loading data
	fakes = pd.read_csv(csv_filenames[0], delimiter = ',', converters={'id':str})
	reals = pd.read_csv(csv_filenames[1], delimiter = ',', converters={'id':str})


	#changing id format
	fakes.id += '-FAKE'
	reals.id += '-REAL'


	# #appending tags
	fakes['Tag'] = ['FAKE' for i in range(fakes.shape[0])]
	reals['Tag'] = ['REAL' for i in range(reals.shape[0])]

	# #setting dataframe index col
	fakes = fakes.rename({'id':'Id'},axis='columns').set_index('Id')
	reals = reals.rename({'id':'Id'},axis='columns').set_index('Id')

	#concatenating dataframes, sorting by index and renaming labels
	df = pd.concat([fakes,reals]).sort_index().rename({'qtd_modals/qtd_verbs':'Uncertainty','(qtd_ind_reference+qtd_group_reference)/qtd_pronouns':'nonImediacy'},axis='columns')
	df = df.reset_index()

	return df.drop('Tag',axis=1)


def countTags(text, tagger):

	wordcount = 0

	#pos tags used by nlpnet
	pos_tags = {'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0, 'VAUX':0}

	#counting sentences
	sentences = len([sentence.strip() for sentence in re.split('[\.\r\n]+',text) if len(sentence) > 0])

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

	# result = list(pos_tags.values())
	result = [0,0]
	#Pausality
	result[0] = pos_tags['PU'] / sentences 
	#Emotiveness
	result[1] = (pos_tags['ADJ'] + pos_tags['ADV'] + pos_tags['ADV-KS'])/(pos_tags['N'] + pos_tags['V'] + pos_tags['NUM'] + pos_tags['NPROP'] + pos_tags['VAUX']) 

	return result


#function that loads the corpus and counts LIWC classes frequencies
def loadMetrics(filenames):

	data = []

	#loading nlpnet	
	tagger = POSTagger(r'var/nlpnet', language='pt')

	# labels = list({'ADJ': 0, 'ADV': 0, 'ADV-KS': 0, 'ART': 0, 'CUR': 0, 'IN': 0, 'KC': 0, 'KS': 0, 'N': 0, 'NPROP': 0, 'NUM': 0, 'PCP': 0, 'PDEN': 0, 'PREP': 0, 'PROADJ': 0, 'PRO-KS': 0, 'PROPESS': 0, 'PROSUB': 0, 'V': 0, 'PU': 0}.keys())
	labels = ['Pausality', 'Emotivity'] #, 'Uncertainty', 'Nonimediatism']

	#loading files
	for filename in filenames:
		with open(filename, encoding='utf8') as f:
			#preprocesses the text read in f using prep()
			#then counts the frequencies using the tagger
			#returns a list with frequencies
			# try:
			freqs = countTags(f.read(),tagger)
			# except:
				# print('Error processing POS with :',filename,flush=True)
				# continue 

			#then appends this list into the data segment of the result dict
			data.append(freqs)

	df = pd.DataFrame(data,columns=labels)
	df_extra_features = loadMetricsCSV(['fakes.csv','reals.csv'])
	df = pd.concat([df,df_extra_features],axis=1).drop('Id',axis=1)
	
	return df