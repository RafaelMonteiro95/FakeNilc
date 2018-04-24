import argparse
import pandas as pd 
import sys
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC

#TODO: add logging system, and remove prints

def parseArguments():
	arg_parser = argparse.ArgumentParser(description='A fake news classifier training system')
	arg_parser.add_argument('dataset_filename', help='path to the file used as dataset')
	arg_parser.add_argument('--n_jobs', help='number of threads to use when cross validating')
	arg_parser.add_argument('-v','--verbose', help='output verbosity.', action='store_true')
	args = arg_parser.parse_args()

	### Parameters
	dataset_filename = args.dataset_filename
	verbose = args.verbose
	if args.n_jobs != 0:
		n_jobs = args.n_jobs
	else:
		n_jobs = 2

	return (dataset_filename, verbose, n_jobs)


def printResults(real, predicts, f=sys.stdout):
	print('Learning curve values:')
	#printing learning curve
	v = len(real)
	s = [int(v*0.2), int(v* 0.4), int(v*0.6), int(v*0.8), v]
	accs = [100*accuracy_score(real[:s[i]], predicts[i]) for i in range(len(predicts))]
	print('00.0000(0%) {0:.4f}(20%) {1:.4f}(40%) {2:.4f}(60%) {3:.4f}(80%) {4:.4f}(100%)'.format(*accs))

	#printing classification report
	print('Classification Report:')
	print(classification_report(real,predicts[4]), file = f)

	#printing confusion matrix
	tn, fp, fn, tp = confusion_matrix(real, predicts[4]).ravel()
	print('Confusion Matrix:', file = f)
	print(' a      b     <--- Classified as', file = f)
	print('{0:5d}  {1:5d}   a = REAL'.format(tp,fp), file = f)
	print('{0:5d}  {1:5d}   b = FAKE'.format(fn,tn), file = f)


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


def main():

	#parsing arguments from command line
	dataset_filename, verbose, n_jobs = parseArguments()
	
	if(verbose):
		print(sys.argv[1:])
		print('verbosity turned on')

	# loading dataset into a pandas dataframe
	df = loadDataset(dataset_filename)

	# getting data and labels from dataframe
	# X = data, y = labels
	X, y = getDatasetValues(df)

	#preparing classifier
	svm = LinearSVC(C=1.0,verbose=verbose)

	#trains the classifier using 5-fold cv and generates an array with results
	#predicts contains the cross validation result for 5 percentiles of the dataset used:
	#20%, 40%, 60%, 80% and 100%
	predicts = predictAndEvaluate(svm, X, y, n_jobs, verbose)

	#printing the result
	printResults(y, predicts)

if __name__ == '__main__':
	main()