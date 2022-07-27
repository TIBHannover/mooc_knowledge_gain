# Feature Extraction
This software realizes the extraction of 386 textual features of slides and transcripts. Syntactic, lexical, structural, temporal, semantic features are extracted. Readability is also determined using various indexes for the files. The semantic features consist of features about sentence embeddings to get similiarity of the containing sentences and lines. The lines of the slides and the sentences of the transcripts are used to calculate those features.
The results are stored as a single csv file.
## Installation
Before the installation of the packages the use on windows requires Microsoft Visual C++ 14.0+.
This can be downloaded under https://visualstudio.microsoft.com/de/visual-cpp-build-tools/ (download and install C++ buildtools with default optional)

The package stanza use CoreNLP to make some taks. CoreNLP is a java-application. For the development Java 8 was used (java version "1.8.0_251")
To install all necessary packages use the command in this directory:
```
pip install -r requirements.txt
```

## Structure
```bash
.
├── corenlp # Contains corenlp. If this folder does not exist it will be automatically downloaded
├── Data # This folder contains the slides, transcripts and the test results
    ├── Slides # This folder contains the slides for the experiments
		├── 1_2a.pdf
		├── ...
		├── 7_3c.pdf
	├── Slides-Processed # contains the extracted text from the pdf-files in reading direction as txt-file
		├── 1_2a.txt
		├── ...
		├── 7_3c.txt	
    ├── Test # This folder contains the results of the tests from Shi et al.
		├──	test.csv
    ├── Transcripts # This folder contains the transcripts for the experiments
		├── 1_2a.srt
		├── ...
		├── 7_3c.srt
├── Features # This folder contains the results of the feature extraction
		├── all_features.csv # only classic features (also contains video_id, person_id for a better overview)
        ├── text_features.csv # subset of all_features.csv, it contains only the text features (also contains video_id, person_id for a better overview)
        ├── multimedia_features.csv # subset of all_features.csv, it contains only the multimedia features (also contains video_id, person_id for a better overview)
		├── slide_embedding.csv # contains the average sentence embeddings for a presentation
		├── transcript_embedding.csv #	contains the average sentence embeddings for a speech transcript
├── wordlists
	├── AoA_51715_words.csv # contains the age of acquistion of words
	├── freq_syll_words.csv # contains the frequency and amount of syllables of words
	├── stopwords.txt # contains stopwords (list from coreNLP)
├── requirements.txt # the necessary packages
├── tenses.py # realises the features about the tenses
├── SortedCollection.py # realises the sorting of lines in the pdf-files
├── readability.py # realises the features about the readability
├── processor.py # realises the feature extraction
├── pre_processor.py # realises preprocessing to extract features
├── main.py # starts of the software
├── files.py # loads and stores files
├── embedding.py # creates the embeddings and specific features about them
├── CompoundWordSplitter.py # Checks if word is a compound word
└── README.md

```
This structure shows the required directories.
Important to mention is that the slides and transcript must have the same name to be correct caluclated. Also this software uses a dimension reduction of embeddings to embeddings with a 16 dimensionality. This reduction uses PCA and can only be used when there is a minimum of 16 datasets.
## Used data
For the results of the experiments several available data have been used.
The slides and transcripts are taken from the edX course [Globally Distributed Software Engineering](https://learning.edx.org/course/course-v1:DelftX+GSE101x+1T2018/home)

The age of acquisition list contains the age of acquisition of over 50.000 words which is provided under this [link](http://crr.ugent.be/archives/806).

The frequency and amount of syllables of words was generated with the help of the [English Lexicon Project](https://elexicon.wustl.edu/query13/query13.html). To get the same file select *Freq_KF* and *NSyll*, and *the complete ELP lexicon* to execute the correct query.  

The stopwords which are used during the calculation of the features are from the coreNLP library. This list can be downloaded on [GitHub](https://github.com/stanfordnlp/CoreNLP/blob/main/data/edu/stanford/nlp/patterns/surface/stopwords.txt).
## How to use
Before starting the software download the previous mentioned data and place it on the corresponding positions as explained in the structure overview. Besides, use the same names as the structure is using.

To run the software and to extract the features from the files use:
```
python main.py
```
On the first start it will take longer because the software downloads the necessary packages for stanza and sentence-transformers to realise their functions.

The three files will be stored in the directory *Features*. *features.csv* contains classic features (also contains video_id, person_id for a better overview), *slide_embedding.csv* contains the average sentence embeddings for a presentation and *transcript_embedding.csv* contains the average sentence embeddings for a speech transcript.

