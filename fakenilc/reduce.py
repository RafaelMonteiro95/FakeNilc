import os
import argparse

import fakenilc.preprocess.reduce as rc


# create a argparser
def prepareArgParser():
	arg_parser = argparse.ArgumentParser(description='A fake news classifier training system')
	arg_parser.add_argument('input_dir', help='path to the folder containing texts to be reduced.')
	arg_parser.add_argument('-o','--output', help='output folder', default='reduced_texts')
	arg_parser.add_argument('-t','--truncate', help='truncates text instead of waiting for end of sentence.', action='store_true')
	arg_parser.add_argument('-v','--verbose', help='output verbosity.', action='store_true')
	return arg_parser


# parses arguments from argparser
def parseArgs(arg_parser):
	args = arg_parser.parse_args()
	news_dir = args.input_dir
	output_dir = args.output
	truncate = args.truncate
	verbose = args.verbose
	return (news_dir, output_dir, truncate, verbose)


def main():
	news_dir, output_dir, truncate, verbose = parseArgs(prepareArgParser())

	# creating dir for storing reduced texts
	os.makedirs(output_dir, exist_ok=True)
	output_dir += '/'
	os.makedirs(output_dir + 'true', exist_ok=True)
	os.makedirs(output_dir + 'fake', exist_ok=True)
	

	#fetching files
	filenames = []
	for true, fake in zip(os.listdir(news_dir + '/true'),os.listdir(news_dir + '/fake')):
		#appends a tuple with a true and a fake file filename
		filenames.append((news_dir + '/true/' + true, news_dir + '/fake/' + fake))

	# reducing files lenght
	for pair in filenames:

		true_filename = pair[0]
		fake_filename = pair[1]
		true_name = true_filename.split('/')[-1]
		fake_name = fake_filename.split('/')[-1]

		#opening files
		with open(pair[0], encoding='utf8') as true:
			with open(pair[1], encoding='utf8') as fake:

				#read both files and reduce the lenght of the biggest
				result = rc.reduce(true.read(),fake.read(), truncate)

				#saves result
				with open(output_dir + 'true/' + true_name,'w', encoding='utf8') as f:
					print(result[0],file=f)
				with open(output_dir + 'fake/' + fake_name,'w', encoding='utf8') as f:
					print(result[1],file=f)


if __name__ == '__main__':
	main()
	