import os
import sys
import argparse
from extractor import liwc, bow, arff, pos
import pandas as pd

def joincsv(filenames, output_filename):

	# reading files
	dfs = [pd.read_csv(filename) for filename in filenames]
	dfr = dfs[0]

	# concatenating dataframes
	for i in range(1,len(dfs)):
		dfr = pd.concat([dfr,dfs[i]],axis=1)

	# writting result
	dfr.to_csv(output_filename, index=False)


if __name__ == '__main__':

	###TODO: PARSE THESE FROM COMMAND LINE
	output_dir = '.'
	output_csv = 'var/csv/'
	news_dir = 'var/texts'
	parameters = []
	join = True
	# join = False
	###END TODO

	# parameters.append('Freq-Full')
	# parameters.append('Freq-2000')
	parameters.append('Freq-5000')
	# parameters.append('Freq-10000')
	# parameters.append('Freq-15000')
	# parameters.append('Freq-20000')
	parameters.append('LIWC')
	parameters.append('POS')
	
	# creating dir for storing temporary csvs
	os.makedirs('var/csv', exist_ok=True)

	# Loading corpus

	# Fetching files from input
	# From the true news folder
	filenames = []
	tags = []
	for filename in os.listdir(news_dir + '/real'):
		filenames.append(news_dir + '/real/' + filename)
		tags.append('REAL')
	# From the fake news folder
	for filename in os.listdir(news_dir + '/fake'):
		filenames.append(news_dir + '/fake/' + filename)
		tags.append('FAKE')


	#preparing features for extraction
	calls = []
	for feature in parameters:
		#extracts POS
		if feature.lower() == 'pos':
			calls.append((pos.loadPos,[filenames, tags, None, False, None]))
		#extracts LIWC tags
		elif feature.lower() == 'liwc':
			calls.append((liwc.loadLiwc,[filenames, tags, None, None, False]))
		#extracts coh-metrix features
		elif feature.lower() == 'coh-metrix':
			#TODO
			raise NotImplementedError('coh-metrix')
		#extracts bag of words representation
		elif feature.lower().split('-')[0] == 'freq':
			if(feature.lower().split('-')[1].lower() == 'full'):
				calls.append((bow.loadCount,[filenames, tags, None, None, None]))
			else:
				calls.append((bow.loadCount,[filenames, tags, int(feature.lower().split('-')[1].lower()), None, None]))
		else:
			raise ValueError(feature + ' is not a valid feature')

	# Extracts each feature described in the calls list
	for parameter,call in zip(parameters, calls):
		print('Extracting', parameter, '...')
		feature_filename = output_csv + parameter.lower()
		
		#if this feature was already extracted, dont need to extract it again
		if(os.path.isfile(feature_filename + '.csv')):
			continue;

		# generates a data dict by calling the correct feature method, with correct parameters
		feature_method = call[0]
		feature_parameters = call[1]
		result = feature_method(*feature_parameters)

		# writes a csv for the extracted feature
		arff.createCSV(result['labels'], result['data'], relation=parameter, filename=feature_filename)

	# Creating a csv with trustworthy tags
	arff.createCSV(['Trustworthy'], tags, relation='Trustworthy', filename=output_csv + 'tags')

	if(join):
		print('Joining csv...')
		csv_filenames = [output_csv+parameter.lower()+'.csv' for parameter in parameters]
		csv_filenames.append(output_csv + 'tags.csv')
		output_filename = output_dir + '-'.join([parameter.lower() for parameter in parameters])
		output_filename += '-tagged'
		joincsv(csv_filenames, output_filename + ".csv")
		print('Result saved to', output_dir + output_filename + ".csv", flush = True)