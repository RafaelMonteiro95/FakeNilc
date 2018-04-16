import argparse
import pandas as pd 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm

def print_cm():
	""

if __name__ == '__main__':

	arg_parser = argparse.ArgumentParser(description='A fake news classifier training system')
	arg_parser.add_argument('dataset_filename', help='path to the file used as dataset')
	arg_parser.add_argument('tags_filename', help='path to the file with dataset instances labels (REAL or FAKE)')
	arg_parser.add_argument('-v','--verbose', help='output verbosity.', action='store_true')
	args = arg_parser.parse_args()

	# exit(0)

	### Parameters
	dataset_filename = args.dataset_filename
	tags_filename = args.tags_filename
	verbose = args.verbose
	if(args.verbose):
		print('verbosity turned on')
	###

	# opening dataset
	# loading dataset into a pandas dataframe
	with open(dataset_filename, encoding='utf8') as features, open(tags_filename, encoding='utf8') as tags:
		df_f = pd.read_csv(features)
		df_t = pd.read_csv(tags)

	# X contain a matrix with dataframe values, y contains a 
	X = df_f.values
	#transposing gives me an array with an array inside it. I just need a single array
	y = df_t.values.transpose()[0]

	print(y[:5])

	X, y = shuffle(X, y)

	print(y[:5])

	# print(X.head())
	# print(y.head())

	#preparing classifier
	classifier = svm.SVC(C=1, verbose=verbose)
	predicted = cross_val_predict(classifier, X, y, cv=5, verbose=verbose, n_jobs=-1)

	#result:
	print(classification_report(y,predicted))
	# cm = confusion_matrix(y,predicted,['Real','Fake'])
	# print_cm(cm,labels)
	tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()

	print('Confusion Matrix:')
	print(' a      b     <--- Classified as')
	print('{0:5d}  {1:5d}   a = REAL'.format(tp,fp))
	print('{0:5d}  {1:5d}   b = FAKE'.format(fn,tn))