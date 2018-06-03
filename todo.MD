# TO DO

## General
Update verbosity system to use logs instead of prints (partially done, need to check preprocessing module files)

## extract.py
Switch from multiple scripts with functions to multiple classes with methods. Create an abstract Feature class with method Load that extracts the required features and labels
Extract some of coh-metrics features using another tool (coh-metrix doesn't work for some texts)
Create interface compatible with scikit-learn


## classify.py
save trained models using pickle
create interface for classification without training a new model (loading best model with pickle) 
create/find a way to get a confiance percentage on SVM classifiers (distance from instance to support vectors or some kind of regression, maybe)


## Perfomance
Improve perfomance on feature extraction and text reduction using parallel computing
Improve mlp training speed using GPU


## Documentation
Add informations about each parameter in readme.MD
keep this document up to date

## Etc
Create a web interface for this project