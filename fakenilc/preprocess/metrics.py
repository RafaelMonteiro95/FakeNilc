# -*- coding: utf-8 -*-

from fakenilc.preprocess import utils
import pandas as pd
import numpy as np
import string
import re

#supress some warnings about type conversion from nlpnet
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	from nlpnet import POSTagger

#Note: this fucntion depends on how the atributes are stored and isn't really bugproof. Rewrite this whenever needed.
def loadMetricsCSV(path):

	#opening datasets
	ni_fake = pd.read_csv(path + 'non_immediacy_fake.csv', delimiter=';', header=None, names=['id','non_immediacy'], converters={'id':str})
	un_fake = pd.read_csv(path + 'uncertainty_fake.csv', delimiter=';', header=None, names=['id','uncertainty'], converters={'id':str})
	ni_real = pd.read_csv(path + 'non_immediacy_true.csv', delimiter=';', header=None, names=['id','non_immediacy'], converters={'id':str})
	un_real = pd.read_csv(path + 'uncertainty_true.csv', delimiter=';', header=None, names=['id','uncertainty'], converters={'id':str})

	reals = pd.merge(ni_real, un_real, on='id')
	fakes = pd.merge(ni_fake, un_fake, on='id')

	#changing id format
	fakes.id += '-FAKE'
	reals.id += '-REAL'

	#appending tags
	fakes['Tag'] = ['FAKE' for i in range(fakes.shape[0])]
	reals['Tag'] = ['REAL' for i in range(reals.shape[0])]

	#setting dataframe index col
	fakes = fakes.rename({'id':'Id'},axis='columns').set_index('Id')
	reals = reals.rename({'id':'Id'},axis='columns').set_index('Id')

	#concatenating dataframes, sorting by index and renaming labels
	df = pd.concat([fakes,reals]).sort_index().rename({'qtd_modals/qtd_verbs':'Uncertainty','(qtd_ind_reference+qtd_group_reference)/qtd_pronouns':'nonImediacy'},axis='columns')
	df = df.reset_index()

	df = df.rename({'non_immediacy':'nonImediacy'},axis='columns')
	df = df.rename({'uncertainty':'Uncertainty'},axis='columns')

	# #returns the resulting dataframe without the tag column
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


def getNonImmediacy(filenames,metrics_path):
	with open(metrics_path + 'metrics.csv', encoding='utf8') as features:
		df = pd.read_csv(features,index_col=0)

	#Pausality,Emotivity,nonImediacy,Uncertainty
	# Dropping the column with tags
	df = df.reset_index()
	df = df.drop('Id',axis=1)
	df = df.drop('Tag',axis=1)
	df = df.drop('Pausality',axis=1)
	df = df.drop('Emotivity',axis=1)
	# df = df.drop('nonImediacy',axis=1)
	df = df.drop('Uncertainty',axis=1)

	return df


def getPausality(filenames,metrics_path):
	with open(metrics_path + 'metrics.csv', encoding='utf8') as features:
		df = pd.read_csv(features,index_col=0)

	#Pausality,Emotivity,nonImediacy,Uncertainty
	# Dropping the column with tags
	df = df.reset_index()
	df = df.drop('Id',axis=1)
	df = df.drop('Tag',axis=1)
	# df = df.drop('Pausality',axis=1)
	df = df.drop('Emotivity',axis=1)
	df = df.drop('nonImediacy',axis=1)
	df = df.drop('Uncertainty',axis=1)

	return df


def getEmotivity(filenames,metrics_path):
	with open(metrics_path + 'metrics.csv', encoding='utf8') as features:
		df = pd.read_csv(features,index_col=0)

	#Pausality,Emotivity,nonImediacy,Uncertainty
	# Dropping the column with tags
	df = df.reset_index()
	df = df.drop('Id',axis=1)
	df = df.drop('Tag',axis=1)
	df = df.drop('Pausality',axis=1)
	# df = df.drop('Emotivity',axis=1)
	df = df.drop('nonImediacy',axis=1)
	df = df.drop('Uncertainty',axis=1)

	return df


def getUncertainty(filenames,metrics_path):
	with open(metrics_path + 'metrics.csv', encoding='utf8') as features:
		df = pd.read_csv(features,index_col=0)

	#Pausality,Emotivity,nonImediacy,Uncertainty
	# Dropping the column with tags
	df = df.reset_index()
	df = df.drop('Id',axis=1)
	df = df.drop('Tag',axis=1)
	df = df.drop('Pausality',axis=1)
	df = df.drop('Emotivity',axis=1)
	df = df.drop('nonImediacy',axis=1)
	# df = df.drop('Uncertainty',axis=1)

	return df


#function that loads the corpus and counts LIWC classes frequencies
def loadMetrics(filenames):

	data = []

	#loading nlpnet	
	tagger = POSTagger(r'var/nlpnet', language='pt')

	labels = ['Pausality', 'Emotivity']

	#loading files
	for filename in filenames:
		with open(filename, encoding='utf8') as f:

			# calculates all frequencies
			freqs = countTags(f.read(),tagger)
			# inserts them into data matrix
			data.append(freqs)

	# turns data matrix into a dataframe
	df = pd.DataFrame(data,columns=labels)
	# loads features that i already have saved in .csv files
	df_extra_features = loadMetricsCSV('var/metrics_csv/')
	# concatenates the two dataframes: the one with features i've extracted, and the one with features i got from the .csv files
	df = pd.concat([df,df_extra_features],axis=1).drop('Id',axis=1)
	
	return df