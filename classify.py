import argparse
import pandas as pd 
import sys
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#TODO: add logging system, and remove prints


def parseArguments():

	choices = ['svc','linearsvc','naive_bayes','randomforest', 'all']


	arg_parser = argparse.ArgumentParser(description='A fake news classifier training system')
	arg_parser.add_argument('dataset_filename', help='path to the file used as dataset')
	arg_parser.add_argument('--n_jobs', help='number of threads to use when cross validating')
	arg_parser.add_argument('-v','--verbose', help='output verbosity.', action='store_true')
	arg_parser.add_argument('-o','--output', help='output filename')
	arg_parser.add_argument('-c','--classifier', help='Specific classifier to be used. If ommited, uses all classifiers', choices=choices)
	args = arg_parser.parse_args()

	### Parameters
	#dataset filename
	dataset_filename = args.dataset_filename
	#verbosity level
	verbose = args.verbose

	#paralelism level used in cross validation
	if args.n_jobs != 0:
		n_jobs = args.n_jobs
	else:
		n_jobs = 2

	#file output. if none, prints result to screen
	if args.output != None:
		output = open(args.output,'w')
	else:
		n_jobs = sys.stdout

	#classifiers used. if 'all' or None, uses all classifiers
	classifier = [LinearSVC(C=1.0),
					  MultinomialNB(),
					  RandomForestClassifier(),
					  MLPClassifier()]
	if args.classifier == 'linearsvc':
		classifier = [classifier[0]]
	elif args.classifier == 'naive_bayes':
		classifier = [classifier[1]]
	elif args.classifier == 'randomforest':
		classifier = [classifier[2]]
	elif args.classifier == 'mlp':
		classifier = [classifier[3]]
	else:
		pass #all classifiers are already set

	return (dataset_filename, verbose, n_jobs, output, classifier)


def loadDataset(dataset_filename):
	with open(dataset_filename, encoding='utf8') as features:
		df = pd.read_csv(features,index_col=0)
	return df


def getDatasetValues(df):
	# Getting the tags column and saving it into y
	y = df.loc[:,'Tag'].tolist()
	# Dropping the column with tags
	df = df.drop('Tag',axis=1)

	# X contain a matrix with dataframe values, y contains a 
	X = df.values

	#shuffling dataset
	X, y = shuffle(X, y)

	return (X, y)


def predictAndEvaluate(classifier, X, y, n_jobs = 2, verbose = False):

	#calculating slices of the dataset
	v = len(y)
	s = [int(v*0.2), int(v* 0.4), int(v*0.6), int(v*0.8), v]

	predicts = [
				cross_val_predict(classifier, X[:s[0]], y[:s[0]], cv=5, verbose=verbose, n_jobs=n_jobs),
				cross_val_predict(classifier, X[:s[1]], y[:s[1]], cv=5, verbose=verbose, n_jobs=n_jobs),
				cross_val_predict(classifier, X[:s[2]], y[:s[2]], cv=5, verbose=verbose, n_jobs=n_jobs),
				cross_val_predict(classifier, X[:s[3]], y[:s[3]], cv=5, verbose=verbose, n_jobs=n_jobs),
				cross_val_predict(classifier, X[:s[4]], y[:s[4]], cv=5, verbose=verbose, n_jobs=n_jobs),
	]

	return predicts


def printResults(real, predicts, f=sys.stdout):
	print('\nLearning curve values:', file = f)
	#printing learning curve
	v = len(real)
	s = [int(v*0.2), int(v* 0.4), int(v*0.6), int(v*0.8), v]
	accs = [100*accuracy_score(real[:s[i]], predicts[i]) for i in range(len(predicts))]
	print('00.0000(0%) {0:.2f}(20%) {1:.2f}(40%) {2:.2f}(60%) {3:.2f}(80%) {4:.2f}(100%)\n'.format(*accs), file = f)

	#printing classification report
	print('Classification Report:', file = f)
	print(classification_report(real,predicts[4]), file = f)

	#printing confusion matrix
	tn, fp, fn, tp = confusion_matrix(real, predicts[4]).ravel()
	print('Confusion Matrix:', file = f)
	print(' a      b     <--- Classified as', file = f)
	print('{0:5d}  {1:5d}   a = REAL'.format(tp,fp), file = f)
	print('{0:5d}  {1:5d}   b = FAKE\n'.format(fn,tn), file = f)


def main():

	#parsing arguments from command line
	dataset_filename, verbose, n_jobs, output, classifier = parseArguments()
	
	if(verbose):
		print(sys.argv[1:], flush=True)
		print('verbosity turned on', flush=True)

	# loading dataset into a pandas dataframe
	df = loadDataset(dataset_filename)

	# getting data and labels from dataframe
	# X = data, y = labels
	X, y = getDatasetValues(df)

	for clf in classifier:
		#trains the classifier using 5-fold cv and generates an array with results
		#predicts contains the cross validation result for 5 percentiles of the dataset used:
		#20%, 40%, 60%, 80% and 100%
		predicts = predictAndEvaluate(clf, X, y, n_jobs, verbose)

		if verbose:
			print('Training ', clf.__class__.__name__, flush=True)

		if verbose:
			print('Saving results for', clf.__class__.__name__, flush=True)

		print('Classifier:', clf.__class__.__name__, file = output)
		#printing the result
		printResults(y, predicts, f = output)
		print('====== ====== ======', file=output)
		print('      =      =      ', file=output)
		print('====== ====== ======', file=output)
		#no need to keep this classifier up
		del clf 

if __name__ == '__main__':
	main()