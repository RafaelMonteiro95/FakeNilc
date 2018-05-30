# -*- coding: utf-8 -*-
import os
import re

def wordcount(string):

	count = 0
	#split string into multiple sentences
	for sentence in re.split('([,\.\n])', string):
		#checks each word in the sentence
		for word in sentence.strip().split():
			# this regex skips punctuations, as I don't consider it words.
			if(re.match('([,\.\n])',word)):
				continue
			# sums 1 to the word count
			count += 1
	return count


def reducestr(str, limit):
	result = []
	count = 0
	#splits the text into sentences
	for sentence in re.split('([\.\n])', str):
		# counts words in each sentence
		for word in sentence.strip().split():
			# this regex skips punctuations, as I don't consider it words.
			if(re.match('([,\.\n])',word)):
				continue
			count += 1
			# result += word + " "
		# if the 
		if count > limit:
			break
		result += sentence
	return ''.join(result)


def reducestr_truncate(str, limit):
	result = []
	count = 0
	#splits the text into sentences
	for word in str.split():
		result += word + " "
		if(re.match('([,\.\n])',word)):
			continue
		# counts words in each sentence
		count += 1
		
		# if the number of words is bigger than the limit 
		if count > limit:
			break

	return ''.join(result)


def reduce(text1,text2, truncate = False):

	# counting number of words in texts
	c1 = wordcount(text1)
	c2 = wordcount(text2)

	# reduces the bigger text in size
	if(c1 > c2):
		text1 = reducestr_truncate(text1,c2) if truncate else reducestr(text1, c2)	
	else:
		text2 = reducestr_truncate(text2,c1) if truncate else reducestr(text2, c1)

	return(text1,text2)