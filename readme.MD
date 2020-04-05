# FakeNilc
A "Fake News" classifier developed in Python3.

This project contains tools used to train classifiers that identify deceptive texts based on syntatic (Part of Speech Tags) and semantic (LIWC word classes, Unigram bag-of-words) linguistic features.

To train and test models, you must use a corpus with real and fake news. You can use the corpus used during the development of this project: https://github.com/roneysco/Fake.br-Corpus.

For more informations on the project, please visit http://nilc-fakenews.herokuapp.com/ or https://sites.google.com/icmc.usp.br/opinando/

If you want to understand better what should be the workflow when using these tools, check the session How it Works

## tl-dr;

Install [Poetry](https://python-poetry.org/docs/#installation) and use the following commands:

```Bash
# clone this repo
git clone https://github.com/RafaelMonteiro95/FakeNilc.git
# install dependecies
poetry install
# clones the repo with texts
git clone https://github.com/roneysco/Fake.br-Corpus.git
# run reduce to balance texts size
poetry run reduce -v -t Fake.br-Corpus/full_texts/
# run extract to extract features from texts
poetry run extract -v reduced_texts --features unigram-binary
# run evaluate to train and test models from the extracted features
poetry run evaluate -c linearsvc -lc 4 -mf 3 -fs 100 --n_jobs 4 -sm unigram-binary.csv -d
```

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. You can find instructions to install it on [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

Once you have poetry running, clone this repository and use the following command to setup a virtual env and install the project dependencies.

```bash
git clone https://github.com/RafaelMonteiro95/FakeNilc.git
poetry install
```

## Usage

There are three available scripts:

1. `reduce.py`: balance text size between real and fake news.
2. `extract.py`: extract attributes from texts and generates a document-attribute table file for various feature sets.
3. `classify.py`: train and evaluate classifiers for given datasets

You can run the scripts using the `poetry run` command, as exemplified in the following sections.

*Note: You might want to clone the [Fake.br corpora](https://github.com/roneysco/Fake.br-Corpus.git) to use the following scripts.
      
      git clone https://github.com/roneysco/Fake.br-Corpus.git

### Reduce.py

#### Description

This script is used to make text sizes balanced. 

      poetry run reduce [-h] [-o OUTPUT] [-t] [-v] input_dir


This script checks each real and fake news pair and truncate the larger one to have a similar size to the smaller one. Once the number of words on the bigger text has exceeded the number of words of the smaller text, this script will:

a). immediately discard the rest of the text; or 

b). discard the rest of the text past the current phrase.

#### Arguments

**Required arguments**
 - `input_dir` : path to folder containing texts to be balanced. Must be a folder containing two other folders: **real** and **fake**, each containing text files named using an integer (`1.txt`, `2.txt`, ...)

 **Optional arguments**
 - `-h, --help` : display a help message and exit
 - `-o OUTPUT, --output OUTPUT` : saves resulting files in OUTPUT folder
 - `-t, --truncate` : truncate text instead of waiting for end of sentence
 - `-v, --verbose` : shows info messages between steps 

 #### Example Usage
 
         poetry run reduce -v -t Fake.br-Corpus/full_texts/

### Extract.py

Extractes features from texts.

      poetry run extract [-h] [-o OUTPUT_DIR]
               [-f {unigram,unigram-binary,all} [{unigram,unigram-binary,all}]]
               [-v] [-d]
               texts_dir

Available features for extraction:
- unigram: Bag of Words representation which counts word frequencies for each document
- unigram-binary: Bag of Words representation which marks word ocurrence for each document

#### Arguments

**Required arguments**
 - `texts_dir` : path to folder containing texts to have its features extracted. Must be a folder containing two other folders: **real** and **fake**, each containing text files named using an integer (`1.txt`, `2.txt`, ...)

 **Optional arguments**
 - `-h, --help` : display a help message and exit
 - `-o OUTPUT_DIR, --output_dir` : folder where resulting csv's will be saved
 - `-d, --debug` : shows debug messages between steps
 - `-v, --verbose` : shows info messages between steps 
 - `-f {unigram, unigram-binary, all, ...}, --features{unigram, unigram-binary, all, ...}` : selects which features will be extracted. Each feature set will result in a different `.csv` file named after the feature set (i.e. `unigram.csv`)

 #### Example Usage

         poetry run extract -v reduced_texts --features unigram-binary

### Evaluate.py

Train and evaluate models using features extracted previously.

      poetry run evaluate [-h] [--n_jobs N_JOBS] [-sm] [-s] [-v] [-d] [-m]
                [-mf MINIMUM_FREQUENCY] [-fs FEATURE_SELECTION]
                [-lc LEARNING_CURVE_STEPS] [-o OUTPUT]
                [-c {svc,linearsvc,naive_bayes,randomforest,all,mlp}]
                dataset_filenames [dataset_filenames ...]

This script trains classifiers from provided .csv datasets(generated using `extract`) and evaluates them using k-fold cross validation. The result is a report showing precision, recall, f1 measure, confusion matrix and avg accuracy for each classifier.

You can train classifiers using a few different algorithms: SVC, LinearSVC, Naive Bayes, Random Forest and Multilayer Perceptron. The algorithms used are implemented by [scikit-learn](https://scikit-learn.org/stable/index.html), and the default parameters are used.

It is also possible to generate avg accuracy values for increasing percentiles of the dataset(which is useful for building a learning curve) using the `-lc` argument.

#### Arguments

**Required arguments**
 - `dataset_filenames` : path to the dataset file generated by `extract`. Multiple files may be provided and they will be evaluated in sequence.

 **Optional arguments**
 - `-h, --help` : display a help message and exit
 - `-o OUTPUT, --output OUTPUT` : output performance report to OUTPUT file
 - `-d, --debug` : shows debug messages between steps
 - `-v, --verbose` : shows info messages between steps
 - `-s, --simple` : simplifies the performance report
 - `-m, --missed` :  outputs the ID of misclassified texts
 - `-sm, --save_model` :  saves a .pikle file with a trained model 
 - `-mf MINIMUM_FREQ, --minimum_frequency MINIMUM_FREQ` : Applies a frequency cut, removing all columns that occur with a minimum frequency under MINIMUM_FREQ. For unigram models only.
 - `-lc STEPS, --learning_curve_steps STEPS` : Writes the average k-fold accuracy using STEPS percentiles of the dataset. For example, STEPS = 4 would result in calculating avg. accuracy with 25%, 50%, 75% and 100% of the records in the dataset.
 - `--n_jobs K` : Uses `K` CPU threads to train models. Defaults to `-1`, which uses all available threads.
 - `-fs K, --feature_selection K` : Selects the best `K` features among the ones provided in dataset. For example, K = 5 would use the 5 most relevant features.
 - `-c {classifiers},  --classifier {classifiers}` : selects classifiers to evaluate. Available options are `svc,linearsvc,naive_bayes,randomforest,mlp or all`. `all` does not include mlp since it takes a considerable time to evaluate. Defaults to `all`. Multiple arguments may be provided

 #### Example Usage

         poetry run evaluate -c linearsvc -lc 2 -mf 3 -fs 100 -sm unigram-binary.csv -d

## How it works

In order to train machine learning models to detect fake news, you must first gather data. 

From 2017 to 2018, we collected 3600 news from trustworthy newspapers, such as G1, Folha and Estadão, and 3600 fake news from shady websites. You can find those news in the [Fake.br corpus](https://github.com/roneysco/Fake.br-Corpus).

In order to guarantee a balanced learning, we must provide similar amount of informations from fake and real news to the machine. In Fake.br corpus, we have paired fake and real news, so that each pair of texts approaches similar topics. Also, using `reduce`, we guarantee that the texts are also similar in size.

Then, we must extract features from the texts. Using `extract`, we can extract a few different feature sets from the texts, from bag of words representations to syntactic and semantic features. The output from this stage is a .csv file with each column being an specific feature and each row being the values of each feature for a given text file.

Finally, we can train and test the extracted attributes with several machine learning algorithms. `evaluate` is responsible for evaluating using K-fold cross validation, which is one of the most used techniques for evaluating ML models. It is also possible to save a .pickle file containing the trained model to use on different apps, like we use on [our website](http://nilc-fakenews.herokuapp.com/).


## Current issues

- There is no script to run evaluations given a trained model.
- Most feature extraction methods are currently disabled due to external dependencies or errors with the Fake.br corpus. They may or may not be enabled in the future.

## Contact

Get in touch with me through my email [rafaelmonteiro95@gmail.com]().
