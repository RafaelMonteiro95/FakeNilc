import argparse
import pandas as pd 
import sys
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC

#TODO: add logging system, and remove prints

def parseArguments():
	arg_parser = argparse.ArgumentParser(description='A fake news classifier training system')
	arg_parser.add_argument('dataset_filename', help='path to the file used as dataset')
	arg_parser.add_argument('-v','--verbose', help='output verbosity.', action='store_true')
	args = arg_parser.parse_args()

	### Parameters
	dataset_filename = args.dataset_filename
	verbose = args.verbose

	return (dataset_filename, verbose)


def printResults(real, predicted, f=sys.stdout):

	#printing classification report
	print(classification_report(real,predicted), file = f)

	#printing confusion matrix
	tn, fp, fn, tp = confusion_matrix(real, predicted).ravel()
	print('Confusion Matrix:', file = f)
	print(' a      b     <--- Classified as', file = f)
	print('{0:5d}  {1:5d}   a = REAL'.format(tp,fp), file = f)
	print('{0:5d}  {1:5d}   b = FAKE'.format(fn,tn), file = f)


def main():

	dataset_filename, verbose = parseArguments()
	
	if(verbose):
		print(sys.argv[1:])
		print('verbosity turned on')

	# loading dataset into a pandas dataframe
	with open(dataset_filename, encoding='utf8') as features:
		df = pd.read_csv(features,index_col=0)

	# Getting the tags column and saving it into y
	y = df.loc[:,'Tag'].tolist()
	# Dropping the column with tags
	df = df.drop('Tag',axis=1)

	# X contain a matrix with dataframe values, y contains a 
	X = df.values

	#shuffling dataset
	X, y = shuffle(X, y)

	#preparing classifier
	classifier = LinearSVC(C=1.0,verbose=verbose)
	predicted = cross_val_predict(classifier, X, y, cv=5, verbose=verbose, n_jobs=2)

	#result:
	printResults(y, predicted)

if __name__ == '__main__':
	main()