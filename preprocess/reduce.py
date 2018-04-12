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
	for sentence in re.split('([,\.\n])', str):
		for word in sentence.strip().split():
			count += 1
		if count > limit:
			break
		result += sentence
	return ''.join(result)

def reduce(text1,text2):

	c1 = wordcount(text1)
	c2 = wordcount(text2)

	if( c1 > c2):
		text1 = reducestr(text1,c2)		
	else:
		text2 = reducestr(text2,c1)

	return(text1,text2)

	#selecting which text will be reduced
	# bigger_text = text2 if (len(text2.split()) > len(text1.split())) else text1

	# #reducing
	# word_count = 0 #auxiliary word count
	# sentences = [] #list that keeps the sentences that will stay in the text

	# #for each sentence in the big text
	# for sentence in re.split('([,\.\n])', bigger_text):

	# 	#looking for valid sentences (those that arent just punctuations)
	# 	if(re.search('[,\.\n]',sentence) == None):
	# 		print(sentence.strip().split())

	# 	#tokenizes sentence and count number of words
	# 	for word in sentence.strip().split():
	# 		word_count += 1

	# 	#if i've included enough words, ill stop including sentences
	# 	if word_count > len(bigger_text.split()):
	# 		break

	# 	sentences.append(sentence)


	# #joins sentences into a resulting text
	# resulting_text1 = '\n'.join(sentences) if (len(text1) > len(text2)) else text1
	# resulting_text2 = '\n'.join(sentences) if (len(text1) < len(text2)) else text2


	# return (resulting_text1, resulting_text2)