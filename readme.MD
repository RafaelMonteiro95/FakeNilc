# FakeNilc
A "Fake News" classifier developed in Python3.

This project contains tools used to train classifiers that identify deceptive texts based on syntatic (Part of Speech Tags) and semantic (LIWC word classes, Unigram bag-of-words) linguistic features.

For this project, a corpora with fake and real texts news was compiled. Each fake news has a real counterpart, extracted from trustworthy pt-br news websites, such as https://g1.globo.com/, https://www.folha.uol.com.br/ or http://www.estadao.com.br/. Each file contains a single news text and is named as `<ID>.txt`, where ID is an integer from 1 to 3600.

More details soon!

## Usage

Available scripts:
1. `reduce.py`: balance text size between real and fake news text files.
2. `extract.py`: extract attributes from texts and generates a document-attribute table file for various feature sets.
3. `classify.py`: train and evaluate classifiers for given datasets

The scripts are more in depth described in the following sections.

### Reduce.py

**Balances texts sizes.**

This script checks each real and fake news pair and truncate the larger one to have a similar size to the smaller one. Once the number of words on the bigger text has exceeded the number of words of the smaller text, this script will:
a). immediately discard the rest of the text; or 
b). discard the rest of the text past the current phrase.

**Parameters:**
TODO.
For more informations, run `reduce.py -h`

### Extract.py

**Extracts attributes from texts**
To use it, you must run `extract.py`. This will extract a set of features from the texts and save them in .csv files. 

Available features for extraction:
- unigram: Bag of Words representation which counts word frequencies for each document
- unigram-binary: Bag of Words representation which marks word ocurrence for each document
- liwc: Normalized frequency for words of each semantic class of the Brazillian Portuguese LIWC 2007 Dictionary.
- pos: Normalized frequency for Part of Speech tags extracted using nlpnet. More info about nlpnet on https://github.com/erickrf/nlpnet 

**Parameters:**
TODO.
For more informations, run `extract.py -h`


### Classify.py

**Trains and evaluate classifiers**

This script trains classifiers from provided .csv datasets(generated using `extract.py`) and evaluates them using k-fold cross validation. The result is a report showing precision, recall, f1 measure, confusion matrix and avg accuracy for each classifier. It is also possible to generate avg accuracy values for increasing percentiles of the dataset(which is useful for building a learning curve). 


**Parameters:**
TODO.
For more informations, run `classify.py`

## Setup
TODO.

But you most likely just need to install all the required packages

### Requirements

**Python Packages**

1. NLTK (make sure run to run NLTK.download() and download its data)
2. Numpy
3. Scikit-Learn
4. nlpnet (for POS features) https://github.com/erickrf/nlpnet

   nlpnet requires h5py and scipy

5. Pandas 

### etc

1. Download our database [Here](https://drive.google.com/open?id=171xE1ZsVpHQIbhPKdKcLsy4aA3r--bbq)
2. Extract it to a folder
3. Edit the parameters in the `.py` files with the correct parameters


## Contact

Get in touch with me through [rafael.augusto.monteiro@usp.br] or [rafaelmonteiro95@gmail.com].