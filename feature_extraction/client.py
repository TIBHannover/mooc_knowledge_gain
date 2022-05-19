import os
from stanza import install_corenlp
from stanza.server import CoreNLPClient


def install():
    """ Installs the files for corenlp if they do not exist
    """
    if not(os.path.isdir("./corenlp")):
        install_corenlp(dir="./corenlp/")


def init_client():
    """ Initialise the client
    :return: client
    """
    install()
    os.environ["CORENLP_HOME"] = "./corenlp/"
    client = CoreNLPClient(annotators=['tokenize', 'pos', 'parse'],
                           timeout=30000, memory='16G',
                           threads=12,
                           be_quiet=True,
                           preload=True,
                           output_format='json',
                           endpoint='http://localhost:9001')  # Change port if it is closed
    return client


def annotate_sentence(client, sentence):
    """ Sends the sentence to the Stanford Corenlp to process it
    :param client: the active client to use
    :param sentence: the sentence which needs to be processed
    :return: results of processing
    """
    return client.annotate(sentence)


def start_client(client):
    """ Starts the client
    :param client: client to start
    """
    client.start()


def stop_client(client):
    """ Stops the client
    :param client: the client to stop
    """
    client.stop()
