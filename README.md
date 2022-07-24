# Predicting Knowledge Gain for MOOC Video Consumption
This is the official repository for the paper "Predicting Knowledge Gain for MOOC Video Consumption" published at the 23rd International Conference for Artificial Intelligence in Education 2022.

The code is divided into three subprojects consisting of feature extraction of source data, feature dataset processing, and Weka evaluation processing.
## Feature Extraction
The feature extraction is used to generate 386 multimodal eatures from the given slides and transcripts to use them for machine learning tasks. 
## Dataset Processing
The dataset processing uses the extracted features to generate the K-Fold-Cross-Validation data splits to make it possible to use them in a clean way on machine learning algorithms.
## Evaluation of Weka results
The evaluation uses the results of each data split calculated by WEKA and calculates the averaged K-Fold-Cross-Validation results.
