import os
import sys
import argparse
from extractor import liwc, bow, arff, pos, metrics
import numpy as np
import pandas as pd

#TODO: use logging instead of printfs


def parseArguments(choices):
	parameters = []

	#parsing command line arguments
	arg_parser = argparse.ArgumentParser(description='A fake news classifier feature extraction system. Extracts selected features and saves it in one or multiple .csv files')
	arg_parser.add_argument('texts_dir', help='path to the folder containing news used as dataset')
	arg_parser.add_argument('output_location', help='path to the output location')
	arg_parser.add_argument('features', help='features to be extracted. If All is selected, then all features will be extracted. Multiple options can be selected.', nargs='+', choices=choices)
	arg_parser.add_argument('-j','--join', help='if resulting csvs should be joined in one single file or not', action='store_true')
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
	if 'all' in [par.lower() for par in args.features]:
		parameters = choices
		parameters.remove('all')
	else:
		for parameter in args.features:
			parameters.append(parameter)

	#verbosity
	verb = args.verbose

	#if we have to join the csvs
	join = args.join

	return (output_csv, news_dir, parameters, verb, join)


def loadCorpus(news_dir):
	# Loading corpus
	ids = []
	filenames = []
	tags = []

	for filename in os.listdir(news_dir + '/real'):
		ids.append(filename.replace('.txt','-REAL'))
		filenames.append(news_dir + '/real/' + filename)
		tags.append('REAL')

	# From the fake news folder
	for filename in os.listdir(news_dir + '/fake'):
		ids.append(filename.replace('.txt','-FAKE'))
		filenames.append(news_dir + '/fake/' + filename)
		tags.append('FAKE')

	ids, filenames, tags = (list(t) for t in zip(*sorted(zip(ids, filenames, tags))))

	ids = pd.DataFrame(ids,columns=['Id'])
	tags = pd.DataFrame(tags,columns=['Tag'])

	return (ids, filenames, tags)


def prepareCalls(parameters, filenames, tags):
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
		elif feature.lower() == 'metrics':
			calls.append((metrics.loadMetrics,[filenames, tags, None, None, False]))
		elif feature.lower() == 'freq-df3':
			calls.append((bow.loadCount,[filenames, tags, None, None, False, 3]))
		elif feature.lower() == 'uncertainty':
			calls.append((None,None))
		#extracts bag of words representation
		elif feature.lower().split('-')[0] == 'freq':
			if(feature.lower().split('-')[1].lower() == 'full'):
				calls.append((bow.loadCount,[filenames, tags, None, None, None]))
			else:
				calls.append((bow.loadCount,[filenames, tags, int(feature.lower().split('-')[1].lower()), None, None]))
		else:
			raise ValueError(feature + ' is not a valid feature')

	return calls


def extractFeatures(parameters, calls, output_csv, ids, tags, verb = True):
	# Extracts each feature described in the calls list
	for parameter,call in zip(parameters, calls):
		
		if verb:
			print('Extracting', parameter, '...',end='',flush=True)

		#if this feature was already extracted, dont need to extract it again
		feature_filename = output_csv + parameter.lower()
		if(os.path.isfile(feature_filename + '.csv')):
			if verb:
				print('csv already exists. next...',flush=True)
			continue;


		#calling the function
		feature_method = call[0]
		feature_parameters = call[1]
		result = feature_method(*feature_parameters)

		#appeding ids and tags to resulting dataframe
		result_df = pd.concat([ids,result,tags],axis=1)
		result_df = result_df.set_index('Id')

		if verb:
			print('done',flush=True)

		if verb:
			print('Creating csv...',end='',flush=True)

		# writes a csv for the extracted feature
		with open(feature_filename + '.csv', 'w', encoding='utf8',) as f:
			result_df.to_csv(f)
		if verb:
			print('done',flush=True)

	if verb:
		print('Extraction Complete',flush=True)


def joincsv(filenames):
	#resulting dataframe
	dfr = pd.read_csv(filenames[0],index_col=0)
	#dataframe that stores the tags
	#saves the tag column on the 1st csv
	tags = tags = dfr.iloc[:,-1]
	dfr = dfr.drop('Tag',axis=1)
	# reading files
	for i in range(1,len(filenames)):
		#loads csv into df
		df = pd.read_csv(filenames[i],index_col=0)
		#removes the tag column
		df = df.drop('Tag',axis=1)
		#concatenate the new dataframe with resulting dataframe
		dfr = pd.concat([dfr,df],axis=1)

	#concatenates the resulting dataframe with the tags dataframe
	dfr = pd.concat([dfr,tags],axis=1)

	return dfr


def joinFeatures(parameters, output_csv):

	#generating a list with .csv files to load dataframes
	csv_filenames = [output_csv+parameter.lower()+'.csv' for parameter in parameters]
	#creating the joined csv filename
	output_filename = output_csv + '-'.join([parameter.lower() for parameter in parameters])
	#joins all csv files into one dataframe
	df = joincsv(csv_filenames)
	#dumps dataframe
	df.to_csv(output_filename + '.csv')


def main():

	choices = ['Freq-Full','Freq-df3','LIWC', 'POS', 'Metrics', 'Uncertainty','all']

	output_csv, news_dir, parameters, verb, join = parseArguments(choices)

	if(verb):
		print(*sys.argv,sep = ' ')

	# creating output dir
	if(output_csv != ''):
		os.makedirs(output_csv, exist_ok=True)

	#loading corpus
	if(verb):
		print('generating filenames list...',end='',flush=True)
	ids, filenames, tags = loadCorpus(news_dir)
	if(verb):
		print('done',flush=True)

	#generating a list with calls to feature extraction methods and their parameters
	if(verb):
		print('generating parameters list...',end='',flush=True)
	calls = prepareCalls(parameters, filenames, tags)
	if(verb):
		print('done',flush=True)

	#extracts all features
	extractFeatures(parameters, calls, output_csv, ids, tags, verb)

	#joins the resulting csvs files into a single one.
	if(join):
		if verb:
			print('Joining csv...',end='',flush=True)
		joinFeatures(parameters, output_csv)
		if verb:
			print('Done',flush=True)


if __name__ == '__main__':

	main()