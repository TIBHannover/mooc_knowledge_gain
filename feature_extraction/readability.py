import math


def flesch_reading_ease(total_words, total_sentences, total_syllables):
    """ Calculates the value of the flesch reading ease index
    :param total_words: amount of words in the text
    :param total_sentences: amount of sentences in the text
    :param total_syllables: amount of syllables in the text
    :return: value of flesch reading ease index
    """
    if not total_sentences or not total_words:  # if there are no words/sentences the value is 0 (default)
        return 0
    initial = 206.835
    a = 1.015
    b = 84.6
    avg_sen_len = total_words / total_sentences  # average sentence length of the text
    avg_syll = total_syllables / total_words  # average number of syllables per word
    return initial - a * avg_sen_len - b * avg_syll


def flesch_kincaid(total_words, total_sentences, total_syllables):
    """ Calculates the value of the flesch reading ease index
    :param total_words: amount of words in the text
    :param total_sentences: amount of sentences in the text
    :param total_syllables: amount of syllables in the text
    :return: value of flesch kincaid index
    """
    if not total_sentences or not total_words:  # if there are no words/sentences the value is 0 (default)
        return 0
    a = 0.39
    b = 11.8
    sub = 15.59
    avg_sen_len = total_words / total_sentences # average sentence length of the text
    avg_syll = total_syllables / total_words  # average number of syllables per word
    return a * avg_sen_len + b * avg_syll - sub


def gunning_fog(total_words, total_sentences, complex_words):
    """ Calculates the value of the gunning fog index
    :param total_words: amount of words in the text
    :param total_sentences: amount of sentences in the text
    :param complex_words: words that have 3 syllables without the suffixes es, ed, ing and no proper nouns
           or compound words
    :return: value of the gunning fog index
    """
    if not total_sentences or not total_words:  # if there are no words/sentences the value is 0 (default)
        return 0
    a = 0.4
    avg_sen_len = total_words / total_sentences  # average sentence length
    return (avg_sen_len + ((complex_words / total_words) * 100)) * a


def smog(total_sentences, total_polysyllables):
    """ Calculates the value of the smog index
    :param total_sentences: amount of sentences in the text
    :param total_polysyllables: amount of words that have 3 or more syllables
    :return: value of smog index
    """
    if not total_sentences:  # if there are no sentences the value is 0 (default)
        return 0
    a = 1.0430
    min_sentences = 30
    b = 3.1291
    return b + a * math.sqrt(total_polysyllables * min_sentences / total_sentences)


# Automated readability index
def ari(total_chars, total_words, total_sentences):
    """ Calculates the value of the ari
    :param total_chars: amount of chars in the text
    :param total_words: amount of words in the text
    :param total_sentences: amount of sentences in the text
    :return: value of the ari
    """
    if not total_sentences or not total_words:  # if there are no words/sentences the value is 0 (default)
        return 0
    a = 0.50
    b = 4.71
    c = 21.43
    avg_word_len = total_chars / total_words
    avg_sen_len = total_words / total_sentences
    return b * avg_sen_len + a * avg_word_len + - c


def avg_hundred(a, b):
    """ calculate the ratio between a and b and calculates the average per hundred
    :param a: value that should show the average per hundred
    :param b: other value that is the dividend
    :return: avg per hundred
    """
    avg = a / b  # average per one
    return avg * 100  # average per hundred


def coleman_liau(total_chars, total_words, total_sentences):
    """ Calculates the value of the coleman liau index
    :param total_chars: amount of chars in the text
    :param total_words: amount of words in the text
    :param total_sentences: amount of sentences in the text
    :return: value of the coleman liau index
    """
    if not total_sentences or not total_words:  # if there are no words/sentences the value is 0 (default)
        return 0
    a = 0.0588
    b = 0.296
    c = 15.8
    return a * avg_hundred(total_chars, total_words) - b * avg_hundred(total_sentences, total_words) - c


def get_speak_tempo(amount_words, time):
    """ Calculates the speaktempo (words per minute)
    :param amount_words: amount of words in the text
    :param time: time that passed
    :return: speaktempo
    """
    minutes = time / 60
    return amount_words / minutes


def difference_tempo(speak_tempo):
    """ Calculates the subtraction from the speaktempo and the default tempo (180 WPM)
    :param speak_tempo: speaktempo of text
    :return: result of subtraction (speaktempo - default tempo)
    """
    wpm = 180  # default amount of words per minute
    return speak_tempo - wpm


def reading_time(amount_words):
    """ Calculates the reading time of an text with the default tempo of 180 WPM
    :param amount_words: amount of words in the text
    :return: reading time in minutes
    """
    wpm = 180  # default amount of words per minute
    return amount_words / wpm


def speaking_difference(subtitle_time, read_time):
    """ Calculates the subtraction from the speaktime and the readtime (transcripts)
    :param subtitle_time: complete time of the subtitles
    :param read_time: reading time of the text
    :return: subtraction (subtitle_time - read_time)
    """
    return subtitle_time - read_time
