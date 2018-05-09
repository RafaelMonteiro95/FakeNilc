# -*- coding: utf-8 -*-
import os
import re

def wordcount(string):

	count = 0
	#split string into multiple sentences
	for sentence in re.split('([,\.\n])', string):
		for word in sentence.strip().split():
			count += 1
	return count

def reducestr(str, limit):
	result = []
	count = 0
	#splits the text into sentences
	for sentence in re.split('([,\.\n])', str):
		# counts words in each sentence
		for word in sentence.strip().split():
			count += 1
		# if the 
		if count > limit:
			break
		result += sentence
	return ''.join(result)

def reduce(text1,text2):

	# counting number of words in texts
	c1 = wordcount(text1)
	c2 = wordcount(text2)

	# reduces the bigger text in size
	if(c1 > c2):
		text1 = reducestr(text1,c2)		
	else:
		text2 = reducestr(text2,c1)

	return(text1,text2)