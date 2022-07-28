import os
import files
import sys
import pre_processor
import client


def start_nlp():
    """" Start stanza and corenlp server to have a preload.
    """
    try:
        sta = pre_processor.start_stanza('en', 'tokenize,mwt,pos,lemma,ner')
        cli = client.init_client()
        client.start_client(cli)
        return sta, cli
    except:
        pre_processor.init_stanza()
        sta = pre_processor.start_stanza('en', 'tokenize,mwt,pos,lemma,ner')
        cli = client.init_client()
        client.start_client(cli)
        return sta, cli


def extract_text():
    """ Extract the text from the pdf slides (convert them to a .txt file) and
    stores them in ./Data/Slides-Processed
    """

    files.remove_files(files.load_files('./Data/Slides-Processed'))  # Removes previous converted files
    data = files.load_files('./Data/Slides')
    for f in data:
        files.process_pdf(f)


def get_features():
    """ Starts the the program. Take all files and calculates the features for each file. After that the features are
    stored in a csv file.
    """
    if not os.path.exists('./Features/'):
        os.makedirs('./Features/')
    if not os.path.exists('./Data/Slides-Processed'):
        os.makedirs('./Data/Slides-Processed')
    sta, cli = start_nlp()
    extract_text()
    slides = files.load_files('./Data/Slides-Processed')
    transcripts = files.load_files('./Data/Transcripts')
    if len(slides) != len(transcripts):
        print("Error: Every video needs slides and transcript. Please try again.")
        sys.exit(1)
    files.process_files(list(zip(slides, transcripts)), sta, cli)


# Start of the program
if __name__ == "__main__":
    get_features()
