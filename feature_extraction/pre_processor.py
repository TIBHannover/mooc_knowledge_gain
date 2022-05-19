import stanza
import srt
import re


def init_stanza():
    """ Initialise stanza for the first time (downloads the important data)
    """
    stanza.download('en')


def start_stanza(lang, processors):
    """ Starts stanza to use it later
    :param lang: The language which stanza should process
    :param processors: The types how stanza should process the text
    :return: Initialized stanza pipeline to process text
    """
    return stanza.Pipeline(lang=lang, processors=processors)


def get_words(sentence, sta):
    """ Calculates the tokens and pos-tags of a sentence
    :param sentence: The sentence which need to be processed
    :param sta: Stanza object which processes the sentence
    :return: Words object that contains tokens and pos-tags
    """
    words = []
    processed = sta(sentence)
    for sentence in processed.sentences:  # Sometimes stanza interprets the sentence as multiple sentences
        words += sentence.words
    return words


def get_seconds(start, end):
    """ Calculates the speech time of a subtitle
    :param start: Start time of subtitle
    :param end: End time of subtitle (finished)
    :return: The speech time of a subtitle
    """
    begin = start.seconds + (start.microseconds / 1000000)
    end = end.seconds + (end.microseconds / 1000000)
    diff = end - begin
    return diff


def split_sentences(text, sta):
    """" Splits sentences into an array and change the apostrophe of every sentences for better processing.
    :param text: text of the transcripts
    :param sta: stanza object to seperate into sentences
    :return: array of sentences (text of transcript splitted into sentences)
    """
    sentences = []
    data = sta(text)
    for sentence in data.sentences:
        # converts apostrophes for better calculations (for example for time tenses)
        sentence.text = convert_apostrophe_nlp(sentence.text, sentence.words)
        sentences.append(sentence)
    return sentences


def concat_colon(sentences):
    """
     Merges two sentence together when the first one ends with :. : is not the end of sentence in many readability
    scales.
    :param sentences: sentences to modify
    :return: modified sentences
    """
    sentences_text = []  # stores the text of every sentence
    was_colon = False
    for sentence in sentences:
        text = sentence.text
        if was_colon:  # if the previous sentence ends with a colon merge it with the current one
            pos = len(sentences_text) - 1
            if not text.endswith(':'):
                was_colon = False
            sentences_text[pos] = sentences_text[pos] + " " + text
        elif text.endswith(':'):
            was_colon = True
            sentences_text.append(text)
        else:
            sentences_text.append(text)
    return sentences_text


def get_srt(text, sta):
    """ Extracts the components of the srt-file (text/sentences, time and amount of subtitles)
    :param text: the complete text of the srt-file (contains subtitles and time)
    :param sta: stanza object to realise sentence separation
    :return: sentences, complete time of the subtitles and amount of them
    """
    if not text:
        return '', 0, 0
    subtitles = list(srt.parse(text))
    time = 0.0
    amount = len(subtitles)
    parts = ''
    for subtitle in subtitles:
        time += get_seconds(subtitle.start, subtitle.end)
        # converts apostrophes for better calculations . Removes linebreak for better merging
        parts += " " + convert_apostrophe(subtitle.content.replace("\n", " "))
    splitted = split_sentences(parts, sta)
    sentences = concat_colon(splitted)
    return sentences, time, amount


def change_s_apostrophe(sentence, words):
    """ Change the parts of a sentence that contains 's. To know how to rewrite 's the method checks which wordtypes
    comes after 's. So a decision can be made
    :param sentence: sentence to modify
    :param words: array with words/tokens and their wordtype from the sentence in correct order
    :return: modified sentence
    """
    contains = False
    for word in words:
        if contains:
            if word.text.endswith("ing") or "JJ" in word.xpos or "RB" in word.xpos or "TO" in word.xpos or \
                    "CD" in word.xpos or "DT" in word.xpos:
                sentence, contains = sentence.replace("'s", " is", 1), False
            elif "VBN" == word.xpos:
                sentence, contains = sentence.replace("'s", " has", 1), False
            elif "NN" in word.xpos:
                pass
            else:
                sentence, contains = sentence.replace("'s", " is", 1), False  # default when no pattern match
        elif "'s" in word.text:
            contains = True
    if contains:
        sentence, contains = sentence.replace("'s", " is", 1), False
    return sentence


def change_d_apostrophe(sentence, words):
    """ Change the parts of a sentence that contains 's. To know how to rewrite 'd the method checks which wordtypes
    comes after 'd. So a decision can be made
    :param sentence: sentence to modify
    :param words: array with words/tokens and their wordtype from the sentence in correct order
    :return: modified sentence
    """
    contains = False
    is_wh = False
    for word in words:
        # print(word.xpos)
        if contains:
            if "VB" == word.xpos:
                sentence, contains = sentence.replace("'d", " would", 1), False
            elif "VBN" == word.xpos:
                sentence, contains = sentence.replace("'d", " had", 1), False
            elif "have" in word.text:
                sentence, contains = sentence.replace("'d", " would", 1), False
            elif "'d" in word.text:
                sentence = sentence.replace("'d", " would", 1)  # could be would or had
            elif "better" in word.text:
                sentence, contains = sentence.replace("'d", " had", 1), False
            elif "rather" in word.text:
                sentence, contains = sentence.replace("'d", " had", 1), False
        elif "'d" in word.text:
            if is_wh:
                sentence = sentence.replace("'d", " did", 1)
                is_wh = False
            else:
                contains = True
        elif "W" == word.xpos[0]:
            is_wh = True
        else:
            is_wh = False
    if contains:
        sentence = sentence.replace("'d", " had", 1)
    return sentence


def convert_apostrophe(sentence):
    """ Converts apostrophes that are always the same (no need for interpretation)
    :param sentence: Sentence that has to be processed
    :return: modified sentence
    """
    sentence = sentence.replace("’", "'")  # the char ’ made errors.
    repl = [("'m", " am"), ("'re", " are"), ("'ve", " have"), ("he's", "he is"), ("she's", "she is"), ("it's", "it is"),
            ("won't", "will not"), ("can't", "can not"), ("'ll", " will"), ("let's", "let us"), ("n't", " not"),
            ("who's", "who is"), ("where's", "where is"), ("how's", "how is"), ("' ", " ")]
    for a, b in repl:
        sentence = re.sub(a, b, sentence, flags=re.IGNORECASE)
    return sentence


def convert_apostrophe_nlp(sentence, words):
    """ Converts apostrophes that needs to be interpreted with NLP
    :param sentence: Sentence that has to be processed
    :param words: Object that contains words and their pos-tag
    :return: modified sentence
    """
    return change_s_apostrophe(change_d_apostrophe(sentence, words), words)

