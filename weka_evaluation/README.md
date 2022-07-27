## WEKA Evaluation
This software takes the calculated results (stored as csv) of weka and averaged it to get the correct
k-fold-cross validation result. This values are stored as csv files.
## Installation
To install all necessary packages use the command in this directory:
```
pip install -r requirements.txt
```
## Structure
```
├── average_result.py
├── requirements.txt
├── generated_results # the new stored results are inside this folder
└── csv # folder that should contain csv files from weka
```
## How to use
Create a *csv* folder and place the WEKA files inside it.
After that run (for 5-fold-cross-validation) :
```
python average_result.py --path ./csv --k_fold 5
```
