import os
import sys

from arffextractor import liwc, bow, arff, pos

if __name__ == '__main__':

	print('Starting...', flush = True)

	print('Creating output directory...',end='', flush = True)
	os.makedirs('output', exist_ok=True)
	print('Done!', flush = True)

	# Loading corpus
	filenames = []
	tags = []

	print('Loading filenames... ',end='', flush = True)

	for filename in os.listdir('input/true'):
		filenames.append('input/true/' + filename)
		filenames.append('input/false/' + filename)
		tags.append('TRUE')
		tags.append('FALSE')

	#for testing purposes
	# filenames = ['false/1.txt', 'true/1.txt']
	# tags = ['FALSE','TRUE']

	print('Done!', flush = True)

	print('Using',int(len(filenames)),'files.',int(len(filenames)/2),'fake and',int(len(filenames)/2),'real news.', flush = True)

	#LIWC
	print('Processing LIWC... ',end='', flush = True)
	liwcdata = liwc.loadLiwc(filenames, tags, total_normalization = False)
	print('Done!\nWriting ARFF... ',end='', flush = True)
	arff.createARFF(liwcdata['labels'],liwcdata['data'],relation='liwc', filename='output/liwc.arff')
	print('Done!', flush = True)

	#BAG OF WORDS - FREQUENCY
	print('Processing BoW... ',end='', flush = True)
	count = bow.loadCount(filenames, tags)
	print('Done!\nWriting ARFF... ',end='', flush = True)
	arff.createARFF(count['labels'],count['data'],relation='bag-of-words-frequency', filename='output/count.arff')
	print('Done!', flush = True)

	#BAG OF WORDS - TF-IDF
	print('Processing BoW Tf-Idf... ',end='', flush = True)
	tfidf = bow.loadTfidf(filenames, tags)
	print('Done!\nWriting ARFF... ',end='', flush = True)
	arff.createARFF(tfidf['labels'],tfidf['data'],relation='bag-of-words-tf-idf', filename='output/tfidf.arff')
	print('Done!', flush = True)

	#BAG OF WORDS - FREQUENCY
	print('Processing BoW with 100 features... ',end='', flush = True)
	count = bow.loadCount(filenames, tags, max_features = 100)
	print('Done!\nWriting ARFF... ',end='', flush = True)
	arff.createARFF(count['labels'],count['data'],relation='bag-of-words-frequency-100', filename='output/count100.arff')
	print('Done!', flush = True)

	#BAG OF WORDS - TF-IDF
	print('Processing BoW Tf-Idf with 100 features... ',end='', flush = True)
	tfidf = bow.loadTfidf(filenames, tags, max_features = 100)
	print('Done!\nWriting ARFF... ',end='', flush = True)
	arff.createARFF(tfidf['labels'],tfidf['data'],relation='bag-of-words-tf-idf-100', filename='output/tfidf100.arff')
	print('Done!', flush = True)

	#POS TAGS
	print('Processing POS Tags... ',end='', flush = True)
	postags = pos.loadPos(filenames, tags, normalize = False)
	print('Done!\nWriting ARFF... ',end='', flush = True)
	arff.createARFF(postags['labels'],postags['data'],relation='POS-tags', filename='output/postags.arff')
	print('Done!', flush = True)

