import os
import sys
import argparse
from extractor import liwc, bow, arff, pos
import pandas as pd

def joincsv(filenames, output_filename):

	# reading files
	#dfs = dataframes
	dfs = [pd.read_csv(filename) for filename in filenames]
	#dfr = resulting dataframe
	dfr = dfs[0]

	# concatenating dataframes
	for i in range(1,len(dfs)):
		dfr = pd.concat([dfr,dfs[i]],axis=1)

	# writting result
	dfr.to_csv(output_filename, index=False)


if __name__ == '__main__':

	choices = ['Freq-Full','LIWC', 'POS', 'all']
	parameters = []

	#parsing command line arguments
	arg_parser = argparse.ArgumentParser(description='A fake news classifier feature extraction system. Extracts selected features and saves it in one or multiple .csv files')
	arg_parser.add_argument('texts_dir', help='path to the folder containing news used as dataset')
	arg_parser.add_argument('output_location', help='path to the output location')
	arg_parser.add_argument('parameters', help='features to be extracted. If All is selected, then all features will be extracted. Multiple options can be selected.', nargs='+', choices=choices)
	arg_parser.add_argument('-j','--join', help='if resulting csvs should be joined in one single file or not', action='store_true')
	arg_parser.add_argument('-t','--tag', help='if resulting csvs should contain tags or not', action='store_true')
	arg_parser.add_argument('-v','--verbose', help='output verbosity', action='store_true')
	args = arg_parser.parse_args()

	#checking for output location
	if args.output_location == '.':
		output_csv = ''
	else:
		output_csv = args.output_location

	#checking for texts location
	if args.texts_dir == '.':
		news_dir = ''
	else:
		news_dir = args.texts_dir

	#appending the selected parameters
	if 'all' in [par.lower() for par in args.parameters]:
		parameters = choices
		parameters.remove('all')
	else:
		for parameter in args.parameters:
			parameters.append(parameter)

	#verbosity
	verb = args.verbose

	#if we have to join the csvs
	join = args.join

	#if we have to include tags on csvs
	tagger = args.tag


	if(verb):
		print('Arguments',flush=True)
		print('input dir:',news_dir,flush=True)
		print('output dir:',output_csv,flush=True)
		print('parameters:',parameters,flush=True)
		print('join:',join,flush=True)

	# exit(0)
	# creating output dir
	if(output_csv != ''):
		os.makedirs(output_csv, exist_ok=True)

	# Loading corpus

	if(verb):
		print('generating filenames list...',end='',flush=True)
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

	if(verb):
		print('done',flush=True)


	if(verb):
		print('generating parameters list...',end='',flush=True)
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

	if(verb):
		print('done',flush=True)

	# Extracts each feature described in the calls list
	for parameter,call in zip(parameters, calls):
		if verb:
			print('Extracting', parameter, '...',end='',flush=True)
		feature_filename = output_csv + parameter.lower()
		
		#if this feature was already extracted, dont need to extract it again
		if(os.path.isfile(feature_filename + '.csv')):
			if verb:
				print('csv already exists. next...',flush=True)
			continue;

		# generates a data dict by calling the correct feature method, with correct parameters
		feature_method = call[0]
		feature_parameters = call[1]
		result = feature_method(*feature_parameters)
		if verb:
			print('done',flush=True)

		if verb:
			print('Creating csv...',end='',flush=True)
		# writes a csv for the extracted feature
		arff.createCSV(result['labels'], result['data'], relation=parameter, filename=feature_filename)
		if verb:
			print('done',end='',flush=True)


	if(verb):
		print('Creating tags csv...',end='',flush=True)
	# Creating a csv with trustworthy tags
	arff.createCSV(['Trustworthy'], tags, relation='Trustworthy', filename=output_csv + 'tags')
	if(verb):
		print('done',flush=True)

	if(join):
		if verb:
			print('Joining csv...')

		csv_filenames = [output_csv+parameter.lower()+'.csv' for parameter in parameters]
		output_filename = output_dir + '-'.join([parameter.lower() for parameter in parameters])
		if(tagged):
			csv_filenames.append(output_csv + 'tags.csv')
			output_filename += '-tagged'

		joincsv(csv_filenames, output_filename + ".csv")

		if verb:
			print('Result saved to', output_dir + output_filename + ".csv", flush = True)