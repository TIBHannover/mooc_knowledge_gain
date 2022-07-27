# Predicting Knowledge Gain for MOOC Video Consumption
This is the official repository for the paper "Predicting Knowledge Gain for MOOC Video Consumption" published at the 23rd International Conference for Artificial Intelligence in Education (AIED) 2022.

The code is divided into three subprojects consisting of feature extraction of source data, feature dataset processing, and Weka evaluation processing.
## Feature Extraction
The feature extraction is used to generate 386 multimodal features from the given slides and transcripts to use them for machine learning tasks. 
## Dataset Processing
The dataset processing uses the extracted features to generate the K-Fold-Cross-Validation data splits to make it possible to use them in a clean way on machine learning algorithms.
## Evaluation of Weka results
The evaluation uses the results of each data split calculated by WEKA and calculates the averaged K-Fold-Cross-Validation results.
## Data
In this folder there is a list with descriptions of all features calculated by the feature extraction.
## Citation
If you use this repository for your research, please cite the following paper:
```
@InProceedings{10.1007/978-3-031-11647-6_92,
author="Otto, Christian
and Stamatakis, Markos
and Hoppe, Anett
and Ewerth, Ralph",
editor="Rodrigo, Maria Mercedes
and Matsuda, Noburu
and Cristea, Alexandra I.
and Dimitrova, Vania",
title="Predicting Knowledge Gain for MOOC Video Consumption",
booktitle="Artificial Intelligence  in Education. Posters and Late Breaking Results, Workshops and Tutorials, Industry and Innovation Tracks, Practitioners' and Doctoral Consortium",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="458--462",
isbn="978-3-031-11647-6"
}
```
Additionally, since the knowledge gain data from Shi et al. is used, please also cite the following paper:
```
@inproceedings{shi2019investigating,
    author = {Shi, Jianwei and Otto, Christian and Hoppe, Anett and Holtz, Peter and Ewerth, Ralph},
    title = {Investigating Correlations of Automatically Extracted Multimodal Features and Lecture Video Quality},
    year = {2019},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    doi = {10.1145/3347451.3356731},
    booktitle = {Proceedings of the 1st International Workshop on Search as Learning with Multimedia Information},
    pages = {11–19},
    numpages = {9},
    location = {Nice, France},
    series = {SALMM '19},
}
```
