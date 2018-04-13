# FakeNilc
A "Fake News" classifier developed in Python3.

## Usage

There are three steps involved in running this program.
i) preprocessing
ii) feature extraction
iii) classifier training

### Preprocessing
To use it, you must run `reduce.py`. This will balance the corpus texts lenght by comparing the number of words of two given texts (one real, one fake). Edit the parameter `news_dir` with the path to the folder containing texts to be reduced and edit `output_dir` with the path were the reduced texts should be saved

### feature extraction
To use it, you must run `extract.py`. This will extract a set of features from the texts and save them in a .csv file. Then, it can join the .csv files generated for classification.

Parameters
Edit these in the `extract.py` file
`output_dir`: where generated result will be saved
`output_csv`: where generated intermediary .csv files will be saved
`news_dir`: folder containing texts to be used for extraction
`parameters`: list with parameters to be extracted. Each parameter is a string with the name of the feature to be extracted. They are: `LIWC`,`POS`,`Coh-Metrix`,`Freq-Full` (for a bag of words representation of all words in the texts), or `Freq-X` where X is the number of the top X words, according to frequency in the texts.
`join`: boolean. If `True`, will generate a tagged .csv file joining all features described in `parameters`. If `False`, does not join or tag any .csv file

### Classifier Training
Currently, this program does not train classifiers automatically. To do so, use your own scikit-learn script or run Weka.