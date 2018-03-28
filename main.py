import os

from arffextractor import liwc, bow, arff, pos, textshrink

if __name__ == '__main__':

	filenames = []
	tags = []

	# parameters used when extracting features
	parameters = {	'LIWC' : [filenames, tags, None, None, False],
					'BoW-Freq' : [filenames, tags, None, None, None],
					'BoW-Tfidf' : [filenames, tags, None, None, None],
					'BoW-Freq-100' : [filenames, tags, 100, None, None],
					'BoW-Tfidf-100' : [filenames, tags, 100, None, None],
					'POS' : [filenames, tags, None, False, None]}

	#methods used when extracting features
	methods = {	'LIWC' : liwc.loadLiwc,
				'BoW-Freq' : bow.loadCount,
				'BoW-Tfidf' : bow.loadTfidf,
				'BoW-Freq-100' : bow.loadCount,
				'BoW-Tfidf-100' : bow.loadTfidf,
				'POS' : pos.loadPos}

	print('Starting...', flush = True)

	print('Creating output directory... ',end='', flush = True)
	os.makedirs('output', exist_ok=True)
	print('Done!', flush = True)

	textshrink.shrinkTexts()

	# Loading corpus

	print('Loading filenames... ',end='', flush = True)

	# Filenames is a list with filenames
	# Tags is a list ordered in the same way as filenames, that contains the corresponding TrustWorthy tag of each file
	for filename in os.listdir('input/true'):
		filenames.append('input/true/' + filename)
		filenames.append('input/false/' + filename)
		tags.append('TRUE')
		tags.append('FALSE')

	# #for testing purposes
	# filenames = ['false/1.txt', 'true/1.txt']
	# tags = ['FALSE','TRUE']

	print('Done!', flush = True)

	print('Using',int(len(filenames)),'files.',int(len(filenames)/2),'fake and',int(len(filenames)/2),'real news.', flush = True)

	for feature in methods:
		
		feature_name = feature
		feature_method = methods[feature]
		feature_parameters = parameters[feature]
		feature_filename = 'output/' + feature.lower() + '.arff'

		print('Processing',feature,'... ',end='', flush = True)
		# generates a data dict by calling the correct feature method, with correct parameters
		data = feature_method(*feature_parameters)

		print('Done!\nWriting ARFF... ',end='', flush = True)

		#creates a .arff file for that feature
		arff.createARFF(data['labels'], data['data'], relation=feature_name, filename=feature_filename)

		print('Done! saved to "' + feature_filename + '"', flush = True)
