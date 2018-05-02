# TO DO

## General
Parse arguments from command line on `reduce.py`
Update verbosity system to use logs instead of prints 

## Feature Extraction
Switch from multiple scripts with functions to multiple classes with methods. Create an abstract Feature class with method Load that extracts the required features and labels
Extract some of coh-metrics features using another tool (coh-metrix doesn't work for some texts)

## Classification
train and save trained models instead of just evaluating their performance with 5-fold cv

## Perfomance
Improve perfomance on feature extraction and text reduction using parallel computing
Improve mlp training speed using GPU

## Documentation
Update readme.MD with correct instructions for CLI arguments of `extract.py` and `classify.py`