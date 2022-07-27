# Dataset Processing
This software uses the calculated features from the [feature extraction](https://github.com/TIBHannover/mooc_knowledge_gain/tree/main/feature_extraction) and generates dataset splits (train set, test set) for k-cross-validation.
In particular, a feature selection is performed to better evaluate the influence of individual features in subsequent experiments.

It is possible to generate two types of experimental data. The first type of experimental data contains
creates datasets with the help of average achieved knowledge gain class per participant that saw a video. The second 
type contains non averaged knowledge gain of a user per video.

The generated files can be used for machine learning experiments in WEKA for example.
## Installation
To install all necessary packages use the command in this directory:
```
pip install -r requirements.txt
```
## Structure
```bash
.
├── features # This folder contains the calculated data from the feature extraction
    ├── all_features.csv
    ├── text_features.csv 
    ├── multimedia_features.csv
    ├── slide_embedding.csv
    ├── transcript_embedding.csv
├── feature_selection
    ├── all # contains dataset splits of non averaged samples (V111 experiment)
        ├── train # contains the generated train data
        ├── test # contains the generated test data
    ├── avg # contains dataset splits of averaged samples (V22 experiment)
        ├── train # contains the generated train data
        ├── test # contains the generated test data
    ├── feature_importance
├── feature_selection.py
├── merge_files.py
├── requirements.txt
└── README.md

```
## How to use
This subproject is divided into two files. The *feature_selection.py* file generates the experimental data and the 
*merge_files.py* merges the non-embedding features with the embedding features for the corresponding k-fold set.
Additionally, the Feature Importance is stored to view the influence of individual features.

This results in datasets without embedding features, datasets with embedding features of the slides, datasets with 
embedding features of the transcripts and datasets only with the respective feature embeddings of the file or 
both in combination (slides and transcripts).
### feature_selection.py
Before running the python file ensure that the calculated feature files are placed inside a features folder as shown in 
the structure illustrated.

The generation of datasets can be realized by different settings.

To create the V111 experiment dataset of our paper run:
```
python feature_selection.py --path ./features/ --method python --feature_importance --drop_column --filter influence
  --k_fold 5
```
The results are stored under:
```
/feature_selection/all/
/feature_selection/feature_importance/
```
For the V22 experiment datatset run:
```
python feature_selection.py --path ./features/ --method avg_python --feature_importance --drop_column --filter influence
  --k_fold 5
```
The results are stored under:
```
/feature_selection/avg/
/feature_selection/feature_importance/
```
### merge_files.py
This file helps to merge the datasplits of the classic features (one dimensional) and the embedding features.
To run this python script for non averaged data execute (k for k-fold was 5 in our experiments):
```
python merge_files.py --folder all --k_fold 5
```
To use averaged data execute:
```
python merge_files.py --folder avg --k_fold 5
```
