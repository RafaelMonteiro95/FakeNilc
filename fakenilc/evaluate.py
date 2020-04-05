import argparse
import sys
import logging
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
import pandas as pd 
import numpy as np
from fakenilc.preprocess import bow

#TODO: add logging system, and remove prints

def prepareArgParser():
	clf_choices = ['svc','linearsvc','naive_bayes','randomforest', 'all', 'mlp']
	

	arg_parser = argparse.ArgumentParser(description='A fake news classifier training system')
	arg_parser.add_argument('dataset_filenames', help='path to the files used as datasets.', nargs='+')
	arg_parser.add_argument('--n_jobs', help='number of threads to use when cross validating', type=int, default=2)
	arg_parser.add_argument('-sm','--save_model', help='if a trained .pickle model should be saved.', action='store_true')
	arg_parser.add_argument('-s','--simple', help='prints simple results.', action='store_true')
	arg_parser.add_argument('-v','--verbose', help='output verbosity.', action='store_true')
	arg_parser.add_argument('-d','--debug', help='output debug messages', action='store_true')
	arg_parser.add_argument('-m','--missed', help='print ids of instances that were incorrectly classified', action='store_true')
	arg_parser.add_argument('-mf','--minimum_frequency', help='minimum frequency for unigrams', type=int, default=1)
	arg_parser.add_argument('-fs','--feature_selection', help='realizes the evaluation of the most relevant features', type=int, default=-1)
	arg_parser.add_argument('-lc','--learning_curve_steps', help='no. of percentages used in learning curve. If -1, the learning curve is not calculated (default)', type=int, default=1)
	arg_parser.add_argument('-o','--output', help='output filename', type=argparse.FileType('w', encoding='UTF-8'), default='-')
	arg_parser.add_argument('-c','--classifier', help='Specific classifier to be used. If ommited, uses all classifiers except for MLPClassifier', choices=clf_choices)
	return arg_parser


def parseArguments():

	arg_parser = prepareArgParser()
	args = arg_parser.parse_args()

	### Parameters
	flags = {}
	#dataset filename
	dataset_filenames = args.dataset_filenames
	#verbosity
	flags['v'] = (args.verbose or args.debug)
	#debug verbosity
	flags['d'] = args.debug
	#print incorrect classifications ids
	flags['m'] = args.missed
	#paralelism level used in cross validation
	flags['n_jobs'] = args.n_jobs if args.n_jobs else 2
	#no. of percentages used in learning curve
	flags['lc'] = args.learning_curve_steps
	#min. freq for unigrams
	flags['mf'] = args.minimum_frequency
	#min. freq for unigrams
	flags['fs'] = args.feature_selection
	#simple output
	flags['s'] = args.simple
	#output file
	output = args.output
	#trained model must be generated and saved
	flags['sm'] = args.save_model

	#classifiers used. if 'all' or None, uses all classifiers
	classifier = [LinearSVC(),
					  MultinomialNB(),
					  RandomForestClassifier()
					  ]
	if args.classifier == 'linearsvc':
		classifier = [classifier[0]]
	elif args.classifier == 'naive_bayes':
		classifier = [classifier[1]]
	elif args.classifier == 'randomforest':
		classifier = [classifier[2]]
	elif args.classifier == 'mlp':
		classifier = [MLPClassifier()]
	else:
		pass #all classifiers except mlp are already set

	return (dataset_filenames, flags, output, classifier)


def loadDatasets(filenames, min_freq):
	logger = logging.getLogger(__name__)

	#resulting dataframe
	logger.info('Opening ' + filenames[0])
	dfr = pd.read_csv(filenames[0],index_col=0)

	#dataframe that stores the tags
	#saves the tag column on the 1st csv
	tags = dfr.iloc[:,-1]
	dfr = dfr.drop('Tag',axis=1)

	#checking if dataframe contains unigram attributes
	if("unigram" in filenames[0]):
		logger.info('Applying frequency cut on dataframe with dimensions ' + str(dfr.values.shape))
		#applies frequency cut
		dfr = bow.removeMinFreqDf(dfr, min_freq)
		logger.info('Resulting dimensions: ' + str(dfr.values.shape))

	# reading files
	for i in range(1,len(filenames)):
		#loads csv into df
		df = pd.read_csv(filenames[i],index_col=0)
			
		#removes the tag column
		df = df.drop('Tag',axis=1)

		#checking if dataframe contains unigram attributes
		if("unigram" in filenames[i]):
			logger.info('Applying frequency cut on dataframe with dimensions ' + str(df.values.shape))
			#applies frequency cut
			df = bow.removeMinFreqDf(df, min_freq)
			logger.info('Resulting dimensions: ' + str(df.values.shape))

		dfr = pd.concat([dfr,df],axis=1)

	#concatenates the resulting dataframe with the tags dataframe
	dfr = pd.concat([dfr,tags],axis=1)

	return dfr


def getDatasetValues(df):
	# Getting the tags column and saving it into y
	y = df.loc[:,'Tag'].tolist()
	# Dropping the column with tags
	df = df.drop('Tag',axis=1)

	# X contain a matrix with dataframe values, y contains a 
	X = df.values

	# Id Contain Indexes
	Id = df.index.values

	#shuffling dataset
	X, y, Id = shuffle(X, y, Id)

	return (X, y, Id)


def predictAndEvaluate(classifier, X, y, dataset_name, lc = 5,  n_jobs = 2, verbose = False, feature_selection = -1, save_model = False):

	logger = logging.getLogger(__name__)

	#calculating slices of the dataset
	s = (np.linspace(0,1,lc+1) * len(y)).astype(np.int)[1:] #creates an array from 0.1 to 1 with 10 evenly spaced items, and multiply by the number of instances of the dataset

	predicts = []
	for val in s:
		logger.info('cross evaluating with '+ str((val/len(y))*100) + '% of corpus')
		if feature_selection > 0:
			predicts.append( cross_val_predict(make_pipeline(SelectKBest(mutual_info_classif,feature_selection),classifier), X[:val], y[:val], cv=5, verbose=verbose, n_jobs=n_jobs) )
		else:
			predicts.append( cross_val_predict(classifier, X[:val], y[:val], cv=5, verbose=verbose, n_jobs=n_jobs) )


	if save_model:
		#generating a filename for pickle file
		model_name = (classifier.__class__.__name__ + '_' + (dataset_name.split('\\')[-1].split('/')[-1].split('.')[0]) + '.pkl').lower()
		logger.info('Trainig model '+ model_name.split('.')[0]+ ' on full dataset')
		#training model
		classifier.fit(X, y)
		#saving trained model using joblib
		joblib.dump(classifier,model_name)


	return predicts


def printResults(classifier, real, predicts, f = sys.stdout):

	logger = logging.getLogger(__name__)

	#printing classification report
	print('Classifier:', classifier, file=f)
	print('Accuracy:', accuracy_score(real,predicts[-1]), file=f)
	print(classification_report(real,predicts[-1]), file = f)

	#printing confusion matrix
	tn, fp, fn, tp = confusion_matrix(real, predicts[-1]).ravel()
	print('Confusion Matrix:', file = f)
	print(' a      b     <--- Classified as', file = f)
	print('{0:5d}  {1:5d}   a = REAL'.format(tp,fp), file = f)
	print('{0:5d}  {1:5d}   b = FAKE\n'.format(fn,tn), file = f)
	print(file=f)

	if len(predicts) > 1:
		p = np.linspace(0,1,len(predicts)+1)[1:] #percentages
		# logger.debug(p)
		s = (p * len(real)).astype(np.int) #dataset sizes
		#calculating accuracy score for each prediction
		scores = [ accuracy_score(real[:s[i]], predicts[i]) for i in range(len(predicts)) ]
		print('Learning curve:',file=f)
		print(scores,file=f)


def printResultsSimple(classifier, real, predicts, f = sys.stdout):

	logger = logging.getLogger(__name__)

	#printing classification report
	print('acc:', accuracy_score(real,predicts[-1]), file = f)
	print(classification_report(real,predicts[-1]), file = f)


def main():

	#parsing arguments from command line
	dataset_filenames, flags, output, classifiers = parseArguments()
	
	#setting verbosity level to python logger
	logging.basicConfig()
	logger = logging.getLogger(__name__)
	if flags['v']: logger.setLevel(logging.INFO) 
	if flags['d']: logger.setLevel(logging.DEBUG) 

	#loading datasets
	dataset_name = '-'.join([name.split('/')[-1].replace('.csv','') for name in dataset_filenames])
	logger.info('Loading dataset ' + dataset_name)
	#loads the dataset into a pandas dataframe
	df = loadDatasets(dataset_filenames, flags['mf'])

	# for each file in the dataset files
	# for dataset_filename in dataset_filenames:
	print('Dataset:', dataset_name, file=output)

	if flags['sm'] and ('unigram' in dataset_name) :
		logger.info('Dumping vocabulary to vocabulary.pkl')
		joblib.dump(list(df),'vocabulary.pkl')

	#split the dataframe in X(data) and y(labels)
	logger.info('Splitting labels and data...')
	X, y, Ids = getDatasetValues(df)


	if flags['fs'] > 0:
		logger.info('Selecting K best features...')
		slct = SelectKBest(mutual_info_classif, flags['fs'])
		slct.fit(X,y)

		sortedList = []
		for i in slct.get_support(indices=True):
			sortedList.append((slct.scores_[i],df.columns[i]))

		print("Best {0} features:".format(flags['fs']),file=output)
		print(sorted(sortedList),file=output)

	predicts = []
	#trains and evaluate each classifier described in classifiers
	for clf in classifiers:

		logger.info('Evaluating ' + clf.__class__.__name__)

		#predicts is a list with lists of labels classified in a 5-fold cross validation evaluation of the classifiers
		#each value in predicts represents a percentage of the dataset used for validation
		#i.e. if predicts contains 3 items, then p[0] used 33% p[1] used 66% and p[2] used 100% of the dataset for evaluation
		predicts = predictAndEvaluate(clf, X, y, dataset_name, flags['lc'] , flags['n_jobs'], flags['v'], flags['fs'], flags['sm'])

		logger.info('Printing Results')
		if flags['s']:
			printResultsSimple(clf.__class__.__name__, y, predicts, f=output)
		else:
			printResults(clf.__class__.__name__, y, predicts, f=output)

		#After evaluating, deletes the used classifier
		del clf
	print('==============',file=output)

	if flags['m']:
		missed = [Ids[i] + ' Classified as ' + predicts[-1][i] + '\n' for i in range(len(y)) if predicts[-1][i] != y[i] ]
		print(*missed, file = output)

	logger.info('Done')


if __name__ == '__main__':
	main()