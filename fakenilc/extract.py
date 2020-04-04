import os
import sys
import argparse
import logging
from fakenilc.preprocess import liwc, bow, pos, syntax, metrics
import numpy as np
import pandas as pd


def parseArguments():

	choices = [
		'unigram',
		'unigram-binary',
		# all these extraction methods arent working correctly
		# 'liwc',
		# 'pos',
		# 'metrics',
		# 'pausality',
		# 'uncertainty',
		# 'emotivity',
		# 'nonimmediacy',
		# 'syntax',
		'all'
		]
	#parsing command line arguments
	arg_parser = argparse.ArgumentParser(description='A text feature extraction system. Extracts selected features and saves it in one or multiple .csv files')
	arg_parser.add_argument('texts_dir', help='path to the folder containing news used as dataset')
	arg_parser.add_argument('-o', '--output_dir', help='path to where the output (generated csvs) should be saved', default='.')
	arg_parser.add_argument('-f', '--features', help='Features to be extracted. Default: all', nargs='+', default=['all'], choices=choices)
	arg_parser.add_argument('-v','--verbose', help='output messages at each step', action='store_true')
	arg_parser.add_argument('-d','--debug', help='output debug messages', action='store_true')
	args = arg_parser.parse_args()

	#checking for texts location
	if args.texts_dir == '.':
		news_dir = ''
	else:
		news_dir = args.texts_dir

	#checking for output location
	if args.output_dir == '.':
		output_dir = ''
	else:
		output_dir = args.output_dir

	#appending the selected parameters
	parameters = []
	if 'all' in [par.lower() for par in args.features]:
		parameters = choices
		#unigrams based on word occurence is default, this removes unigrams based on word frequency
		parameters.remove('unigram')
		#since 'metrics' depends on external extraction of non-immediacy and uncertainty, they also aren't extracted by default
		# parameters.remove('metrics')
		# parameters.remove('pausality')
		# parameters.remove('uncertainty')
		# parameters.remove('emotivity')
		# parameters.remove('nonimmediacy')
		#since syntax depends on external dependencies, I'm removing it from all
		# parameters.remove('syntax')
		parameters.remove('all')
	else:
		for parameter in args.features:
			parameters.append(parameter)

	#verbosity
	verb = args.verbose

	#debugging output
	debug = args.debug

	return (output_dir, news_dir, parameters, verb, debug)


def loadCorpus(news_dir):
	# Loading corpus
	ids = []
	filenames = []
	tags = []


	for filename in os.listdir(news_dir + '/true'):
		ids.append(filename.replace('.txt','-REAL'))
		filenames.append(news_dir + '/true/' + filename)
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


def prepareCalls(parameters, filenames, tags, output_dir):
	#preparing features for extraction
	calls = []
	for feature in parameters:
		#extracts POS
		if feature.lower() == 'pos':
			#loadPos(filenames):
			calls.append((pos.loadPos,[filenames]))
		#extracts LIWC tags
		elif feature.lower() == 'liwc':
			# loadLiwc(filenames):
			calls.append((liwc.loadLiwc,[filenames]))
		#extracts metrics features
		elif feature.lower() == 'metrics':
			# loadMetrics(filenames):
			calls.append((metrics.loadMetrics,[filenames]))
		#extracts unigrams
		elif feature.lower() == 'unigram':
			# loadCount(filenames, min_freq = 1, binary = False, normalize = True)
			calls.append((bow.loadCount,[filenames]))
		elif feature.lower() == 'unigram-binary':
			# loadCount(filenames, min_freq = 1, binary = False, normalize = True)
			calls.append((bow.loadCount,[filenames,1, True, False]))
		elif feature.lower() == 'uncertainty':
			calls.append((metrics.getUncertainty,[filenames, output_dir]))
		elif feature.lower() == 'pausality':
			calls.append((metrics.getPausality,[filenames, output_dir]))
		elif feature.lower() == 'nonimmediacy':
			calls.append((metrics.getNonImmediacy,[filenames, output_dir]))
		elif feature.lower() == 'emotivity':
			calls.append((metrics.getEmotivity,[filenames, output_dir]))
		elif feature.lower() == 'syntax':
			calls.append((syntax.loadSyntax,[filenames]))
		else:
			raise ValueError(feature + ' is not a valid feature')

	return calls


def extractFeatures(parameters, calls, output_csv, ids, tags, verb = True):
	logger = logging.getLogger(__name__)

	# Extracts each feature described in the calls list
	for parameter,call in zip(parameters, calls):
		
		logger.info('Extracting '+ str(parameter))

		#if this feature was already extracted, dont need to extract it again
		feature_filename = output_csv + parameter.lower()
		if(os.path.isfile(feature_filename + '.csv')):
			logger.info('csv already exists. Skipping this extraction')
			continue;

		#calling the function
		feature_method = call[0]
		feature_parameters = call[1]
		result = feature_method(*feature_parameters)

		#appeding ids and tags to resulting dataframe
		result_df = pd.concat([ids,result,tags],axis=1)
		result_df = result_df.set_index('Id')

		logger.info('done. Creating csv')

		# writes a csv for the extracted feature
		with open(feature_filename + '.csv', 'w', encoding='utf8',) as f:
			result_df.to_csv(f)
		logger.info('done')

	logger.info('Extraction Complete')


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

	output_csv, news_dir, parameters, verb, join = parseArguments()

	# logger setup
	logging.basicConfig()
	logger = logging.getLogger(__name__)
	# if verbosity is on, set logger to info level
	if verb: logger.setLevel(logging.INFO) 

	logger.debug(str(sys.argv))

	# creating output dir
	if(output_csv != ''):
		os.makedirs(output_csv, exist_ok=True)

	#loading corpus
	logger.info('generating filenames list')
	ids, filenames, tags = loadCorpus(news_dir)
	logger.info('done')

	#generating a list with calls to feature extraction methods and their parameters
	logger.info('generating parameters list')
	calls = prepareCalls(parameters, filenames, tags, output_csv)
	logger.info('done')

	#extracts all features
	extractFeatures(parameters, calls, output_csv, ids, tags, verb)

	#joins the resulting csvs files into a single one.
	# if(join):
	# 	logger.info('joining csv')
	# 	joinFeatures(parameters, output_csv)
	# 	if verb:
	# 		logger.info('done')


if __name__ == '__main__':

	main()