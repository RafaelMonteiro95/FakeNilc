from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import Tree
from statistics import median
import logging
import pandas as pd
import subprocess
import pickle


def calculate_attributes(parsed_text):
	# Prods contains each production rule and the number of occurences it has
	prods = {}
	# heights contains the heights for each sentence syntax tree
	heights = []
	sents = []
	for row in parsed_text.split('\n'):
		if(row != '' and row != 'SENTENCE_SKIPPED_OR_UNPARSABLE'):
			try:
				sents.append(Tree.fromstring(row))
			except:
				try:
					#sometimes a tree can have an extra bracket; this tries to parse it without the last bracket
					sents.append(Tree.fromstring(row[:-1]))
				except:
					#if it is still wrong, just skip it
					try:
						sents.append(Tree.fromstring(row + ')'))
					except:
						print(row)
	for tree in sents:
		heights.append(tree.height())
		for production in tree.productions():
			if production.is_nonlexical():
				if(not production in prods):
					prods[production] = 1
				else:
					prods[production] += 1
	# If i have an empty height
	if len(heights) == 0:
		print('Empty height', flush=True)
		heights.append(10)
	return (heights, {str(key) : prods[key] for key in prods })


def remove_skipped_sentences(in_filename,out_filename):
	with open(in_filename, encoding = 'utf-8') as f1:
		with open(out_filename,'w', encoding = 'utf8') as f2:
			for line in f1.read().split('\n'):
				if(line == "SENTENCE_SKIPPED_OR_UNPARSABLE"): continue
				print(line,file=f2)


def parse_text(in_filename):
	# opening new file
	# starts the parser process, and prints the output to the new file f
	return subprocess.run(['java','-Xmx4g','-cp','var/stanford-parser-2010-11-30/stanford-parser.jar','edu.stanford.nlp.parser.lexparser.LexicalizedParser', '-maxLength', '100' ,'-tokenized','-sentences','newline','-outputFormat','oneline','-uwModel','edu.stanford.nlp.parser.lexparser.BaseUnknownWordModel','var/cintil.ser',in_filename], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='windows-1252').stdout
	



def tokenize_text(text, out_filename):
	# Expands these words. Extracted from https://pt.wiktionary.org/wiki/Ap%C3%AAndice:Combina%C3%A7%C3%B5es_e_contra%C3%A7%C3%B5es_do_portugu%C3%AAs
	expansions = {'à': 'a a', 'às': 'a as', 'ao': 'a o', 'aos': 'a os', 'do': 'de o', 'da': 'de a', 'dos': 'de os', 'das': 'de as', 'no': 'em o', 'na': 'em a', 'nos': 'em os', 'nas': 'em as', 'pro': 'para o', 'pra': 'para a', 'pros': 'para os', 'pras': 'para as', 'pelo': 'por o', 'pela': 'por a', 'pelos': 'por os', 'pelas': 'por as'	, 'cum': 'com um', 'dum': 'de um', 'duns': 'de uns', 'duma': 'de uma', 'dumas': 'de umas', 'num': 'em um', 'nuns': 'em uns', 'numa': 'em uma', 'numas': 'em umas', 'prum': 'para um', 'pruns': 'para uns', 'pruma': 'para uma', 'prumas': 'para umas', 'comigo': 'com mim', 'contigo': 'com ti', 'consigo': 'com si', 'conosco': 'com nós', 'convosco': 'com vós', 'dele': 'de ele', 'dela': 'de ela', 'deles': 'de eles', 'delas': 'de elas', 'nele': 'em ele', 'nela': 'em ela', 'neles': 'em eles', 'nelas': 'em elas', 'àquele': 'a aquele', 'àquela': 'a aquela', 'àqueles': 'a aqueles', 'àquelas': 'a aquelas', 'àquilo': 'a aquilo', 'deste': 'de este', 'desta': 'de esta', 'destes': 'de estes', 'destas': 'de estas', 'disto': 'de isto', 'desse': 'de esse', 'dessa': 'de essa', 'desses': 'de esses', 'dessas': 'de essas', 'disso': 'de isso', 'daquele': 'de aquele', 'daquela': 'de aquela', 'daqueles': 'de aqueles', 'daquelas': 'de aquelas', 'daquilo': 'de aquilo', 'neste': 'em este', 'nesta': 'em esta', 'nestes': 'em estes', 'nestas': 'em estas', 'nisto': 'em isto', 'nesse': 'em esse', 'nessa': 'em essa', 'nesses': 'em esses', 'nessas': 'em essas', 'nisso': 'em isso', 'naquele': 'em aquele', 'naquela': 'em aquela', 'naqueles': 'em aqueles', 'naquelas': 'em aquelas', 'naquilo': 'em aquilo', 'doutro': 'de outro', 'doutra': 'de outra', 'doutros': 'de outros', 'doutras': 'de outras', 'aonde': 'a onde', 'daqui': 'de aqui', 'daí': 'de aí', 'dali': 'de ali', 'daquém': 'de aquém', 'dalém': 'de além', 'donde': 'de onde', 'pronde': 'para onde', 'pelaí': 'per aí'}
	# Opening files
	with open(out_filename,'w',encoding='utf8') as f2:
		# tokenizing sentence
		for sent in sent_tokenize(text,language='portuguese'):
			if(len(sent)) == 0: continue
			# tokenizing words in sentence
			for token in word_tokenize(sent,language='portuguese'):
				# expanding tokens
				if(token.lower() in expansions):
					token = expansions[token.lower()]
				# saving token to file
				print(token,end=' ', file=f2)
			# printing newline to file
			print('',file=f2)


def calculate_metrics(text):
	tokenize_text(text,'temp1.txt')
	parsed_text = parse_text('temp1.txt')
	data = calculate_attributes(parsed_text)
	# calculating median tree height
	data[1]['MaxHeight'] = float(max(data[0]))
	data[1]['MedianHeight'] = float(median(data[0]))
	return data[1]


def vectorize(text, scaler = None, labels = None):

	if not labels:
		with open('var/syntax_vocab.pkl', 'rb') as f:
			labels = pickle.load(f)

	res = calculate_metrics(text)

	df = pd.DataFrame([res],columns=labels).fillna(0)

	if scaler:
		return pd.DataFrame(scaler.transform(df), columns = labels)

	return df


#function that loads the corpus and counts LIWC classes frequencies
def loadSyntax(filenames):
	logger = logging.getLogger('__main__')

	data = []
	filenames_size = len(filenames)
	for filename, count in zip(filenames, range(1,filenames_size+1)):

		logger.info('File: ' + filename + ' ' + str(count) + '/' + str(filenames_size))

		with open(filename,encoding='utf8') as f:
			data.append(calculate_metrics(f.read()))
		with open('dump_data.pkl','wb') as f:
			pickle.dump(data,f)

	return pd.DataFrame(data).fillna(0)