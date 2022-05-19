import re

import syllapy
import CompoundWordSplitter
import pre_processor
import readability
import tenses
import csv


def count_syllables(csv_file, word):
    """ Counts syllables from a word. The algorithm checks a list if it contains the word or not. If it contains the
        word then the amount of syllables is returned from the list. Else the amount of syllables is calculated
        manually.
    :param csv_file: the file which contains words and syllables
    :param word: the word where the syllables have to be calculated
    :return: Amount of syllables from the word
    """
    words = word.split('-')  # split words to calculate better syllables for each one
    all_syllables = 0
    for w in words:
        syllables = get_amount_syllable(csv_file, w)  # check list for syllables
        if not syllables:  # if value is 0: the word was not in the list
            all_syllables += syllapy.count(w)  # Calculate the syllables without the list
        else:
            all_syllables += syllables
    return all_syllables


def calculate_ngrams(tokens, size):
    """ Calculates n-grams for the given tokens.
    :param tokens: the tokens which should be n-grams
    :param size: it defines the size of an n-gram, for example 2 is a 2-gram
    :return: Matrix with n-grams
    """
    if not size:  # when size is 0 there are no n-grams
        return []
    ngrams = []
    for i in range(len(tokens)):
        ngram = []
        for z in range(i, size + i, 1):
            if z == len(tokens):
                break
            ngram.append(tokens[z])
        if len(ngram) < size:  # n-grams lower than the size are not valid
            break
        ngrams.append(ngram)
    return ngrams


def remove_duplicates(elements):
    """ Removes duplicates from a list
    :param elements: the list which may contain duplicates
    :return: list without duplicates
    """
    # Defines a a dictionary with keys from the list (keys are unique. There are no duplicates).
    dictionary = dict.fromkeys(elements)
    return list(dictionary)


def remove_suffix(word):
    """ Removes the common suffixes (ing, ed, es) of a word.
    :param word: the word which need to be manipulated
    :return: the manipulated word (word with removed suffix)
    """
    if word.endswith('ing'):
        return word[:-len('ing')]
    elif word.endswith('ed'):
        return word[:-len('ed')]
    elif word.endswith('es'):
        return word[:-len('es')]
    return word


def contains_alpha(word):
    """ Checks if word has an alphabet char
    :param word: word which need to be checked
    :return: True if there is an alphabet char else False
    """
    for c in word:
        if c.isalpha():
            return True
    return False


def contains_hyphen(word):
    """ Checks if word has a hyphen
    :param word: the word which must be checked
    :return: True when word contains a hyphen, False when word doesn't contain a hyphen
    """
    if '‐' in word:
        return True
    return False


def is_compound(word):
    """ Checks if the word is a compound of words (for example sunflower. sunflower is the compound of sun and flower.
    :param word: The word which has to be checked
    :return: True when the word is a compound of words else False
    """
    '''
    CompoundWordSplitter splits the word into compounds. If the length of the return is smaller 2 then the word is not
    a compound of word. 
    '''
    if len(CompoundWordSplitter.split(word, "en_en")) <= 1:
        return False
    return True


def is_proper_noun(pos):
    """ Checks if word is a proper noun or not. A proper noun has the pos-tag 'NNP' or 'NNPS'.
    :param pos: the pos which says what type of word the word is.
    :return: True when pos is NNP else False
    """
    if "NNP" in pos:  # is proper noun when pos tag is NNP (singluar) or NNPS (plural)
        return True
    return False


def ratio(a, b):
    """ Calculates the ratio between a and b
    :param a: the value which has to be checked how many % it is in b
    :param b: the value that contains the 100%
    :return: the ratio
    """
    if not b:  # if there is no b (0) a should be stay a. (b should never be 0)
        return a
    else:
        return a / b  # ratio


def get_freq_word(csv_file, word):
    """
    :param csv_file: the csv-file that contains the frequencies of words
    :param word: the specific word to check the frequency for
    :return: the average frequency of the word
    """
    words = word.split('‐')  # Word can consist of more than one word
    freq = 0.0
    for w in words:  # calculate the frequency for every word
        csv_file.seek(1)  # go to start of file again
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            if row[0].lower() == w.lower():
                freq += float(row[1].replace(",", ""))
    return freq / len(words)  # calculate average frequency


def get_amount_syllable(csv_file, word):
    """
    :param csv_file: file that contains syllables for words
    :param word: the specific word to get the syllables for
    :return: the amount of syllables or 0 when the list does not contain the word
    """
    csv_file.seek(1)  # go to start of file again (ignore header)
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        if row[0].lower() == word.lower():
            if row[2] != '#':
                return float(row[2])
            else:
                return 0  # default value to show that there is no value
    return 0  # default value to show that there is no value


def get_age_acquisition(csv_file, word):
    """ Calculate the age of acquisition of a word. If the word consists of more than one word then this function will
    calculate every age of acquisition for each partial word.
    :param csv_file: The file which has the age of acquisition values
    :param word: The word that have to be checked
    :return: The sum of all calculated values
    """
    words = word.split('‐')  # Word can consist of more than one word
    values = 0.0
    for w in words:
        stored = False  # If value of a word is added to values no need to add a default value!
        csv_file.seek(1)  # go to start of file again (ignore header)
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if row[0].lower() == w.lower():  # need to check if word is same. upper and lowercase don't matter
                stored = True
                if not row[10] == "NA":
                    values += float(row[10].replace(",", "."))  # Stores the calculated value from the csv-file
                else:  # no known value, use default value
                    values += 10.36  # average value as default
        if not stored:  # word was not in list
            values += 10.36  # average value as default
    return values / len(words)  # average age of the complete word


class Calculator:
    """ Class that is used for calculating all features
    """

    def __init__(self):
        self.sentences = []  # Array of all sentences/lines
        self.tokens = []  # Array of tokens
        self.lemmas = []  # Array of lemmas
        self.amount_sentences = 0  # Amount of sentences/lines
        self.amount_tokens = 0  # counter for tokens (words)
        self.amount_freq_tokens = 0  # Amount of tokens which can be used for frequency calculation
        self.amount_age_tokens = 0  # Amount of tokens which can be used for age acquisition calculation
        self.amount_unique_tokens = 0  # Amount of unique tokens (words)
        self.ratio_unique_tokens = 0  # Ratio of unique tokens to all tokens
        self.amount_unique_lemmas = 0  # Amount of unique lemmas
        self.ratio_unique_lemmas = 0  # Ratio of unique lemmas to all tokens
        # Attributes for tokens
        self.sum_tok_len = 0
        self.min_tok_len = float('inf')
        self.avg_tok_len = 0
        self.max_tok_len = 0
        self.amount_freq_tok = 0  # All frequences of words added together
        self.avg_freq_tok = 0  # Average frequence of an word
        # Attributes for ngrams
        self.amount_threegrams = 0
        self.avg_threegrams = 0  # Average of threegrams per sentence
        self.amount_fourgrams = 0
        self.avg_fourgrams = 0  # Average of fourgrams per sentence
        # Attributes for sentences
        self.min_sen_len = float('inf')  # minimum amount of words for a sentence
        self.avg_sen_len = 0  # average amount of words for a sentence
        self.max_sen_len = 0  # maximum amount of words for a sentence
        self.min_sen_char_amount = float('inf')  # minimum amount of chars for a sentence
        self.avg_sen_char_amount = 0  # average amount of chars for a sentence
        self.max_sen_char_amount = 0  # maximum amount of chars for a sentence
        # Attributs for syllables
        self.amount_syl = 0  # amount of syllables
        self.amount_one_syl = 0  # amount of words that have only one syllable
        self.amount_two_syl = 0  # amount of words that have two syllables
        self.amount_psyl = 0  # amount of syllables greater than 2
        self.amount_hard = 0  # amount of hard words (over 2 syllables, no compound word, no Proper Noun)
        self.avg_syl = 0  # Average of syllables per word
        self.ratio_one_syl = 0  # Ratio between one syllable words and total words
        self.ratio_two_syl = 0  # Ratio between two syllable words and total words
        self.ratio_psyl = 0  # Ratio between syllables greater than 2 and total words
        self.ratio_hard = 0  # Ratio between hard words and total words
        # Attributes for age of acquistion
        self.sum_age = 0  # Age of acquisition (all values together)
        self.min_age = float('inf')  # Minimum value of the age of acquisition of a word
        self.avg_age = 0  # Average value of the age of acquisition
        self.max_age = 0  # Maximum value of the age of acquisition of a word
        # slides specific
        self.pages = 0  # Amount of pages
        self.sum_lines = 0  # Number of lines
        self.min_lines = float('inf')  # Minimum number of lines per page
        self.avg_lines = 0  # Average number of lines per page
        self.max_lines = 0  # maximum number of lines per page
        self.sum_words_page = 0  # Amount of words of a page
        self.min_words_page = float('inf')  # Minimum number of words per page
        self.avg_words_page = 0  # Average number of words per page
        self.max_words_page = 0  # Maximum number of words per page
        # Readability
        self.flesch_ease = 0  # Flesch Reading Ease
        self.flesch_kin = 0  # Flesch Kincaid
        self.gunning_fog = 0  # Gunning Fog
        self.smog = 0  # SMOG
        self.ari = 0  # Automated Readability Index
        self.coleman = 0  # Coleman Liauf
        self.read_time = 0  # Time to read text
        # Transcript only
        self.speak_time = 0  # Speak time of the video
        self.speak_difference = 0  # Difference between speak and reading time
        self.amount_subtitles = 0  # Amount of subtitles
        # Word-types
        self.amount_adj = 0  # Amount of adjectives
        self.avg_adj = 0  # Average amount of adjectives per sentence
        self.ratio_adj = 0  # Ratio of adjectives to total words
        self.amount_adposition = 0  # Amount of adpositions (pre- and posposition)
        self.avg_adposition = 0  # Average amount of adpositions per sentence
        self.ratio_adposition = 0  # Ratio of adpositions (pre- and posposition) to total words
        self.amount_noun = 0  # Amount of nouns
        self.avg_noun = 0  # Average amount of nouns per sentence
        self.ratio_noun = 0  # Ratio of nouns to total words
        self.amount_pronoun = 0  # Amount of pronouns
        self.avg_pronoun = 0  # Average amount of pronouns per sentence
        self.ratio_pronoun = 0  # Ratio of pronouns to total words
        self.ratio_pronoun_noun = 0  # Ratio of pronouns to total words
        self.amount_verb = 0  # Amount of all verbs
        self.avg_verb = 0  # Average amount of verbs per sentence
        self.ratio_verb = 0  # Ratio of verbs to total words
        self.amount_main_verb = 0  # Amount of all main verbs (no auxiliaries)
        self.avg_main_verb = 0  # Average amount of main verbs per sentence
        self.ratio_main_verb = 0  # Ratio of main verbs to total words
        self.amount_auxiliary = 0  # Amount of auxiliaries
        self.avg_auxiliary = 0  # Average amount of auxiliaries per sentence
        self.ratio_auxiliary = 0  # Ratio of auxiliaries to total words
        self.amount_adverb = 0  # Amount of adverbs
        self.avg_adverb = 0  # Average amount of adverbs per sentence
        self.ratio_adverb = 0  # Ratio of adverbs to total words
        self.amount_coordinate_conj = 0  # Amount of coordinate conjunctions
        self.avg_coordinate_conj = 0  # Average amount of coordinate conjunctions per sentence
        self.ratio_coordinate_conj = 0  # Ratio of coordinate conjunctions to total words
        self.amount_determiner = 0  # Amount of determiners
        self.avg_determiner = 0  # Average amount of determiners per sentence
        self.ratio_determiner = 0  # Ratio of determiners to total words
        self.amount_interjection = 0  # Amount of interjections
        self.avg_interjection = 0  # Average amount of interjections per sentence
        self.ratio_interjection = 0  # Ratio of interjections to total words
        self.amount_num = 0  # Amount of numbers (written as words)
        self.avg_num = 0  # Average amount of numbers (written as words) per sentence
        self.ratio_num = 0  # Ratio of numbers (written as words) to total words
        self.amount_particle = 0  # Amount of particles
        self.avg_particle = 0  # Average amount of particles per sentence
        self.ratio_particle = 0  # Ratio of particles to total words
        self.amount_subord_conjunction = 0  # Amount of subordinates conjunctions
        self.avg_subord_conjunction = 0  # Average amount of subordinates conjunctions per sentence
        self.ratio_subord_conjunction = 0  # Ratio of subordinates conjunctions to total words
        self.amount_foreign_word = 0  # Amount of foreign words
        self.avg_foreign_word = 0  # Average amount of foreign words per sentence
        self.ratio_foreign_word = 0  # Ratio of foreign words to total words

        self.amount_content_word = 0  # Amount of content words
        self.avg_content_word = 0  # Average amount of content words per sentence
        self.ratio_content_word = 0  # Ratio of content words to total words
        self.amount_function_word = 0  # Amount of function words
        self.avg_function_word = 0  # Average amount of function words per sentence
        self.ratio_function_word = 0  # Ratio of function words to total words

        self.amount_filtered = 0  # Amount of tokens that were filtered.
        self.avg_filtered = 0  # Ratio of function filtered words to total words
        self.ratio_filtered = 0  # Ratio of filtered words to unfiltered ones

    def check_min_sen(self, sen_len):
        """ Checks if the current sentence has the minimum length (minimum number of words/tokens)
        :param sen_len: the length of the current sentence
        """
        if sen_len < self.min_sen_len:
            self.min_sen_len = sen_len

    def check_max_sen(self, sen_len):
        """ Checks if the current sentence has the maximum length (maximum number of words/tokens)
        :param sen_len: the length of the current sentence
        """
        if sen_len > self.max_sen_len:
            self.max_sen_len = sen_len

    def check_min_sen_chars_amount(self, chars):
        """ Checks if the current sentence has the minimum amount of chars
        :param chars: the amount of chars
        """
        if chars < self.min_sen_char_amount:
            self.min_sen_char_amount = chars

    def check_max_sen_chars_amount(self, chars):
        """ Checks if the current sentence has the maximum amount of chars
        :param chars: the amount of chars
        """
        if chars > self.max_sen_char_amount:
            self.max_sen_char_amount = chars

    def process_sen(self, sen_len, chars):
        """ Process amount of words/tokens and chars of the current sentence to calculate the min/max values of them
        :param sen_len: amount of words/tokens
        :param chars: amount of chars
        """
        self.check_min_sen(sen_len)
        self.check_max_sen(sen_len)
        self.check_min_sen_chars_amount(chars)
        self.check_max_sen_chars_amount(chars)

    def calculate_avg_sen(self):
        """ Calculates the average sentence length and the average amount of chars per sentence
        """
        if len(self.sentences):
            self.avg_sen_len = self.amount_tokens / len(self.sentences)
            self.avg_sen_char_amount = self.sum_tok_len / len(self.sentences)

    def check_min_tok(self, tok_len):
        """ Checks if the current token has the minimum amount of chars
        :param tok_len: amount of chars
        """
        if tok_len < self.min_tok_len:
            self.min_tok_len = tok_len

    def check_max_tok(self, tok_len):
        """ Checks if the current token has the maximum amount of chars
        :param tok_len: amount of chars
        """
        if tok_len > self.max_tok_len:
            self.max_tok_len = tok_len

    def process_tok(self, tok_len):
        """ Adds the token length to a variable and checks the minimum/maximum length
        :param tok_len: amount of chars in the word/token
        """
        self.sum_tok_len += tok_len
        self.check_min_tok(tok_len)
        self.check_max_tok(tok_len)

    def calculate_avg_tok(self):
        """ Calculates the average length of a word/token (amount of chars)
        """
        if len(self.tokens):
            self.avg_tok_len = self.sum_tok_len / len(self.tokens)

    def check_min_line(self, amount_line):
        """ Checks if the current slide has the minimum amount of slides
        :param amount_line: Amount of lines
        """
        if amount_line < self.min_lines:
            self.min_lines = amount_line

    def check_max_line(self, amount_line):
        """ Checks if the current slide has the maximum amount of slides
        :param amount_line: Amount of lines
        """
        if amount_line > self.max_lines:
            self.max_lines = amount_line

    def calculate_avg_line(self):
        """  Calculates the average amount of lines per slide
        """
        if self.pages:
            self.avg_lines = self.sum_lines / self.pages

    def check_min_words(self):
        """ Checks if the amount of words/tokens of the current slide has the minimum number. if it has the minimum
        number, it will be set as minimum
        """
        if self.sum_words_page < self.min_words_page:
            self.min_words_page = self.sum_words_page

    def check_max_words(self):
        """ Checks if the amount of words/tokens of the current slide has the maximum number. If it has the minimum
        number, it will be set as maximum
        """
        if self.sum_words_page > self.max_words_page:
            self.max_words_page = self.sum_words_page

    def calculate_avg_words(self):
        """ Calculates the average amount of words/tokens per slide
        """
        if self.pages:
            self.avg_words_page = self.amount_tokens / self.pages

    def calculate_threegrams(self, tokens):
        """ Calculates the amount of trigrams of a list of tokens and adds the amount to a variable
        :param tokens: a list of tokens
        """
        if len(tokens):
            self.amount_threegrams += len(calculate_ngrams(tokens, 3))

    def calculate_fourgrams(self, tokens):
        """ Calculates the amount of tetragrams of a list of tokens and adds the amount to a variable
        :param tokens: a list of tokens
        """
        if len(tokens):
            self.amount_fourgrams += len(calculate_ngrams(tokens, 4))

    def calculate_avg_ngrams(self):
        """ Calculates the average amount of trigrams/fourgrams per sentence
        """
        self.avg_threegrams = self.amount_threegrams / self.amount_sentences
        self.avg_fourgrams = self.amount_fourgrams / self.amount_sentences

    def change_inf(self):
        """ Changes every min value which still have the infinite value to 0. The infinitive value was only there to
        calculate the correct min value (f. e. 0 as initial value would cause bugs because higher min values are
        bigger than 0 (so the value 0 does not change)
        """
        if self.min_lines == float('inf'):
            self.min_lines = 0
        if self.min_sen_char_amount == float('inf'):
            self.min_sen_char_amount = 0
        if self.min_words_page == float('inf'):
            self.min_words_page = 0
        if self.min_tok_len == float('inf'):
            self.min_tok_len = 0
        if self.min_sen_len == float('inf'):
            self.min_sen_len = 0
        if self.min_age == float('inf'):
            self.min_age = 0

    def process_page(self, amount_line):
        """ Calculates word and line specific features of a page (a page represents one slide)
        :param amount_line: amount of lines of the page
        """
        # Words
        self.check_min_words()
        self.check_max_words()
        self.sum_words_page = 0
        # Lines
        self.sum_lines += amount_line
        self.check_min_line(amount_line)
        self.check_max_line(amount_line)

    def unique_words(self):
        """ Calculates the amount of unique tokens and unique lemmas. Also it calculates the ratio of them to all
        words/tokens
        """
        filtered_tokens = remove_duplicates(self.tokens)
        filtered_lemmas = remove_duplicates(self.lemmas)
        self.amount_unique_tokens = len(filtered_tokens)
        self.amount_unique_lemmas = len(filtered_lemmas)
        self.ratio_unique_tokens = len(filtered_tokens) / self.amount_tokens
        self.ratio_unique_lemmas = len(filtered_lemmas) / self.amount_tokens

    def calculate_freq(self, properties, word):
        """ Calculate the frequency of a word and adds it to a variable
        :param properties: csv-file that contains frequencies of words
        :param word: the current word
        """
        self.amount_freq_tok += get_freq_word(properties, word)

    def calculate_avg_freq(self):
        """ Calculates the average frequency of a word
        """
        self.avg_freq_tok = self.amount_freq_tok / self.amount_freq_tokens

    def calculate_syllables(self, csv_file, word, pos):
        """  Calculates the syllables of a word. This adds the amount of syllables to a variable and checks whether the
        word is a one-syllable, two-syllable, polysyllable oder hard word. The number of the correct wordclass will be
        incremented
        :param csv_file: cs-file that contains the amount of syllables of words
        :param word: the current word/token to check
        :param pos: the Part-of-Speech Tag of the word/token
        """
        syllables = count_syllables(csv_file, word)
        self.amount_syl += syllables
        if syllables >= 3:  # polysyllable words have 3 or more syllables
            self.amount_psyl += 1
            syllables = count_syllables(csv_file, remove_suffix(word))
            ''' hard words have three or more syllables.  Also they are no proper nouns
            and compound words
            '''
            if syllables >= 3 and not (contains_hyphen(word) or is_compound(word) or is_proper_noun(pos)):
                self.amount_hard += 1
        elif syllables == 2:  # word has two syllables
            self.amount_two_syl += 1
        else:  # word has one syllable
            self.amount_one_syl += 1

    def calculate_relation_syllables(self):
        """ Calculates the average amount of syllables per word and the ratio of the different syllabletypes to all
        words/tokens
        """
        self.avg_syl = self.amount_syl / self.amount_tokens
        self.ratio_one_syl = self.amount_one_syl / self.amount_tokens
        self.ratio_two_syl = self.amount_two_syl / self.amount_tokens
        self.ratio_psyl = self.amount_psyl / self.amount_tokens
        self.ratio_hard = self.amount_hard / self.amount_tokens

    def calculate_age(self, csv_file, word):
        """ Calculates the age of acquistion of a word and checks if this age is the current minimum/maximum value.
        Also it adds the age of the word to a variable
        :param csv_file: the csv-file that contains the age of words
        :param word: the current word
        """
        age = get_age_acquisition(csv_file, word)
        self.calculate_min_age(age)
        self.calculate_max_age(age)
        self.sum_age += age

    def calculate_min_age(self, age):
        """ Checks if an age is the minimum age at the moment. If it is the minimum it will be stored as minimum.
        :param age: age of acquisition of the actually word
        """
        if self.min_age > age:
            self.min_age = age

    def calculate_avg_age(self):
        """ Calculates the average age per word/token
        """
        self.avg_age = self.sum_age / self.amount_age_tokens

    def calculate_max_age(self, age):
        """ Checks if an age is the maximum age at the moment. If it is the maximum it will be stored as minimum.
        :param age: age of acquisition of the actually word
        """
        if self.max_age < age:
            self.max_age = age

    def calculate_speech_diff(self):
        """ Calculates the difference between the real speaking time vs. the calculated reading time. (transcript)
        """
        self.speak_time = self.speak_time / 60
        self.speak_difference = readability.speaking_difference(self.speak_time, self.read_time)

    def calculate_word_types(self, counter):
        """ Calculates the amount, average per sentence and ratio to all words/tokens for every wordtype
        :param counter: counter-object that contains the values of the wordtypes
        """
        self.amount_adj = counter.get_amount_adj()
        self.avg_adj = self.amount_adj / self.amount_sentences
        self.ratio_adj = ratio(counter.get_amount_adj(), self.amount_tokens)
        self.amount_adposition = counter.get_amount_adposition()
        self.avg_adposition = self.amount_adposition / self.amount_sentences
        self.ratio_adposition = ratio(counter.get_amount_adposition(), self.amount_tokens)
        self.amount_noun = counter.get_amount_noun()
        self.avg_noun = self.amount_noun / self.amount_sentences
        self.ratio_noun = ratio(counter.get_amount_noun(), self.amount_tokens)
        self.amount_pronoun = counter.get_amount_pronoun()
        self.avg_pronoun = self.amount_pronoun / self.amount_sentences
        self.ratio_pronoun = ratio(counter.get_amount_pronoun(), self.amount_tokens)
        self.ratio_pronoun_noun = ratio(counter.get_amount_pronoun(), counter.get_amount_noun())
        self.amount_verb = counter.get_amount_verb()
        self.avg_verb = self.amount_verb / self.amount_sentences
        self.ratio_verb = ratio(counter.get_amount_verb(), self.amount_tokens)
        self.amount_main_verb = counter.get_amount_main_verb()
        self.avg_main_verb = self.amount_main_verb / self.amount_sentences
        self.ratio_main_verb = ratio(counter.get_amount_main_verb(), self.amount_tokens)
        self.amount_auxiliary = counter.get_amount_auxiliary()
        self.avg_auxiliary = self.amount_auxiliary / self.amount_sentences
        self.ratio_auxiliary = ratio(counter.get_amount_auxiliary(), self.amount_tokens)
        self.amount_adverb = counter.get_amount_adverb()
        self.avg_adverb = self.amount_adverb / self.amount_sentences
        self.ratio_adverb = ratio(counter.get_amount_adverb(), self.amount_tokens)
        self.amount_coordinate_conj = counter.get_amount_coordinate_conj()
        self.avg_coordinate_conj = self.amount_coordinate_conj / self.amount_sentences
        self.ratio_coordinate_conj = ratio(counter.get_amount_coordinate_conj(), self.amount_tokens)
        self.amount_determiner = counter.get_amount_determiner()
        self.avg_determiner = self.amount_determiner / self.amount_sentences
        self.ratio_determiner = ratio(counter.get_amount_determiner(), self.amount_tokens)
        self.amount_interjection = counter.get_amount_interjection()
        self.avg_interjection = self.amount_interjection / self.amount_sentences
        self.ratio_interjection = ratio(counter.get_amount_interjection(), self.amount_tokens)
        self.amount_num = counter.get_amount_num()
        self.avg_num = self.amount_num / self.amount_sentences
        self.ratio_num = ratio(counter.get_amount_num(), self.amount_tokens)
        self.amount_particle = counter.get_amount_particle()
        self.avg_particle = self.amount_particle / self.amount_sentences
        self.ratio_particle = ratio(counter.get_amount_particle(), self.amount_tokens)
        self.amount_subord_conjunction = counter.get_amount_subord_conjuction()
        self.avg_subord_conjunction = self.amount_subord_conjunction / self.amount_sentences
        self.ratio_subord_conjunction = ratio(counter.get_amount_subord_conjuction(), self.amount_tokens)
        self.amount_foreign_word = counter.get_amount_foreign_word()
        self.avg_foreign_word = self.amount_foreign_word / self.amount_sentences
        self.ratio_foreign_word = ratio(counter.get_amount_foreign_word(), self.amount_tokens)
        self.amount_content_word = counter.get_amount_content_word()
        self.avg_content_word = self.amount_content_word / self.amount_sentences
        self.ratio_content_word = ratio(counter.get_amount_content_word(), self.amount_tokens)
        self.amount_function_word = counter.get_amount_function_word()
        self.avg_function_word = self.amount_function_word / self.amount_sentences
        self.ratio_function_word = ratio(counter.get_amount_function_word(), self.amount_tokens)
        self.amount_filtered = counter.get_amount_filtered()
        self.avg_filtered = self.amount_filtered / self.amount_sentences
        self.ratio_filtered = ratio(counter.get_amount_filtered(), self.amount_tokens)

    def process_readability(self):
        """ Calculates the result of the readability scales.
        """
        self.flesch_ease = readability.flesch_reading_ease(self.amount_tokens, self.amount_sentences, self.amount_syl)
        self.flesch_kin = readability.flesch_kincaid(self.amount_tokens, self.amount_sentences, self.amount_syl)
        self.gunning_fog = readability.gunning_fog(self.amount_tokens, self.amount_sentences, self.amount_hard)
        self.smog = readability.smog(self.amount_sentences, self.amount_psyl)
        self.ari = readability.ari(self.sum_tok_len, self.amount_tokens, self.amount_sentences)
        self.coleman = readability.coleman_liau(self.sum_tok_len, self.amount_tokens, self.amount_sentences)
        self.read_time = readability.reading_time(self.amount_tokens)

    def process_tokens(self, text, sta, is_line, counter, properties, age, stopwords):
        """ Process the tokens of a sentence/line to get the token specific features
        :param text: sentence/line that contains the tokens
        :param sta: stanza object for NLP tasks
        :param is_line: true = text is a line, false = text is a sentence
        :param counter: counter-object for wordtypes/phrases features
        :param properties: csv-file that contains the frequency and amount of syllables of words
        :param age: csv-file that contains the age of acquisition of words
        :param stopwords: the list that contains stopwords
        :return: the modified sentence/line for embedding features,
        """
        data = sta(text)
        is_question = False  # Checks if sentence is question (contains "?")
        context = 0  # counter for real words
        chars = 0  # counter for the chars of the real words
        tokens = []  # Array of unfiltered tokens
        tense = []  # An array of an array of tuples of words and their pos-tag. (only verbs)
        current_tense = []  # An array of tuples of words and their pos-tag of the current sub-sentence
        embedding = ""  # modified sentence for embedding calculation
        non_words_tags = ["PUNCT", "SYM", "X"]  # Tokens that has no meaning for the sentence
        verb_tags = ["VB", "VBP", "VBZ", "VBD", "VBG", "VBN", "MD", "TO"]  # Verb tenses to get tense form
        for sentence in data.sentences:  # Sometimes stanza interprets the sentence as multiple sentences
            for word in sentence.words:
                # print("upos: " + word.upos + " " + "xpos: " + word.xpos + " " + "word: " + word.text)
                xpos = word.xpos
                upos = word.upos
                wordtext = word.text
                lemma = word.lemma
                '''
                Check if token should be calculated in result or not. The word can be a symbol. Most symbols should be
                filtered for example •. But one important symbol for the meaning of the sentence is a mass. The mass
                defines the meaning of a number (30 % , 5 € etc.) 
                '''
                if upos not in non_words_tags or xpos == "NN":
                    tokens.append(wordtext)
                    context += 1
                    chars += len(wordtext)
                    if xpos in verb_tags:
                        current_tense.append((wordtext.lower(), xpos))
                    if is_line:  # calculate line feature if text is a line
                        self.sum_words_page += 1
                    if embedding:  # if modified sentence has content, add whitespace and after that the token
                        embedding += " " + wordtext
                    else:  # add start of modified sentence
                        embedding += wordtext
                    self.tokens.append(wordtext.lower())  # make text lower to check later if it is unique
                    self.lemmas.append(lemma.lower())
                    self.amount_tokens += 1
                    self.process_tok(len(wordtext))
                    self.calculate_syllables(properties, word.text, xpos)
                    # counts the amount of masses
                    if xpos == "NN" and not re.search("[a-zA-Z]", wordtext):
                        counter.inc_mass()
                    ''' calculates the frequency of a token if it is not a number (f. e. 3, for three the frequency will 
                    be calculated) and not  a mass/ unknown word  (f. e. $. A mass has the xpos 'NN' and 
                    does not contain an alphabetic. Unknown words are a mixture of alphabets and other characters)
                    '''
                    if not is_proper_noun(xpos) and wordtext not in stopwords and \
                            ((upos == "NUM" or xpos == "NN") and wordtext.isalpha()) or \
                            (upos != "NUM" and xpos != "NN"):
                        self.calculate_freq(properties, wordtext)
                        self.amount_freq_tokens += 1
                    # calculate age of acquistion
                    if ((upos == "NUM" or xpos == "NN") and wordtext.isalpha()) or (upos != "NUM" and xpos != "NN"):
                        self.calculate_age(age, wordtext)
                        self.amount_age_tokens += 1
                elif wordtext == "?":  # the sentence/line is a question when it contains a '?'
                    is_question = True
                # for better time predicition accurray split parts of sentence
                elif wordtext in [',', ':'] and len(current_tense):
                    tense.append(current_tense)
                    current_tense = []
                counter.store_pos(xpos)
                counter.store_upos(upos)
            if is_question:
                counter.inc_question()
            else:
                counter.inc_statement()
        self.process_sen(context, chars)
        if len(current_tense):  # if the filtered sentence/line has a minimum of 1 token use it for calculations
            tense.append(current_tense)
        self.calculate_threegrams(tokens)
        self.calculate_fourgrams(tokens)
        return embedding, tense, counter

    def process_lines(self, lines, cli, sta, counter, properties, age, stopwords):
        """ Process the lines of a pdf to calculate all the features
        :param lines: the lines of a pdf
        :param cli: client to use the coreNLP
        :param sta: stanza-object for NLP tasks
        :param counter: counter-object for wordtypes and phrasetypes features
        :param properties: the csv-file that contains the frequency and amount of syllables of words
        :param age: the csv-file that contains the age of acquistion of words
        :param stopwords: the list that contains the stopwords (txt file)
        :return: the features for the lines and an array of lines
        """
        time = tenses.Tenses()
        amount_lines = 0
        for line in lines:
            line = line.decode("utf-8")
            if "Starting next page:" in line:
                if self.pages:
                    self.process_page(amount_lines)
                self.pages += 1
                amount_lines = 0
            else:
                amount_lines += 1
                line = pre_processor.convert_apostrophe(line)
                words = pre_processor.get_words(line, sta)
                line = pre_processor.convert_apostrophe_nlp(line, words)
                counter.get_phrases(cli, line)
                embedding, tense, counter = self.process_tokens(line, sta, True, counter, properties, age, stopwords)
                self.sentences.append(embedding)
                time.process_tenses(tense)  # Verb tenses features
        self.process_page(amount_lines)  # stores the values of the last page
        self.amount_sentences = len(self.sentences)
        counter.calculate_avg_sentence_parts(self.amount_sentences)
        self.change_inf()
        self.unique_words()
        self.process_readability()
        self.calculate_avg_age()
        self.calculate_avg_line()
        self.calculate_avg_sen()
        self.calculate_avg_tok()
        self.calculate_word_types(counter)
        self.calculate_avg_freq()
        self.calculate_avg_ngrams()
        self.calculate_avg_words()
        self.calculate_relation_syllables()
        return self.get_features_line() + counter.get_features() + time.get_features(), self.sentences

    def process_sentences(self, text, cli, sta, counter, properties, age, stopwords):
        """
        :param text: the text of a srt-file
        :param cli: client to use the coreNLP
        :param sta: stanza-object for NLP tasks
        :param counter: counter-object for wordtypes and phrasetypes features
        :param properties: the csv-file that contains the frequency and amount of syllables of words
        :param age: the csv-file that contains the age of acquistion of words
        :param stopwords: the list that contains the stopwords (txt file)
        :return: the features for the sentences and an array of sentences
        """
        time = tenses.Tenses()
        # extracts the components of the srt-file
        text, self.speak_time, self.amount_subtitles = pre_processor.get_srt(text.decode('utf-8'), sta)
        for sentence in text:  # iterate through the text (sentence for sentence)
            # print(sentence)
            counter.get_phrases(cli, sentence)
            embedding, tense, counter = self.process_tokens(sentence, sta, False, counter, properties, age, stopwords)
            self.sentences.append(embedding)
            time.process_tenses(tense)
        self.amount_sentences = len(self.sentences)
        counter.calculate_avg_sentence_parts(self.amount_sentences)
        self.change_inf()
        self.unique_words()
        self.process_readability()
        self.calculate_avg_age()
        self.calculate_avg_sen()
        self.calculate_avg_tok()
        self.calculate_word_types(counter)
        self.calculate_avg_freq()
        self.calculate_avg_ngrams()
        self.calculate_speech_diff()
        self.calculate_relation_syllables()
        return self.get_features_sentence() + counter.get_features() + time.get_features(), self.sentences

    def __get_features(self, ignore):
        """ Collects all features and returns them
        :param ignore: attributes that are not part of the features of the specific file
        :return: returns the features of this class
        """
        features = []
        for elem in self.__dict__.items():
            if elem[0] not in ignore:
                features.append(round(elem[1], 6))
        return features

    def get_features_line(self):
        """ Collects the features for a pdf and returns them
        :return: returns the features for a pdf
        """
        return self.__get_features(["sentences", "amount_sentences", "tokens", "amount_freq_tokens",
                                    "amount_age_tokens", "lemmas",
                                    "amount_freq_tok", "speak_time", "amount_subtitles",
                                    "speak_difference", "sum_words_page", "sum_age", "amount_threegrams",
                                    "amount_fourgrams"])

    def get_features_sentence(self):
        """ Collects the features for a srt-file and returns them
        :return: returns the features for a srt-file
        """
        ignore = ["sentences", "tokens", "amount_freq_tokens", "amount_age_tokens", "lemmas", "amount_freq_tok",
                  "pages", "sum_lines", "min_lines", "avg_lines", "max_lines", "sum_words_page",
                  "min_words_page", "avg_words_page", "max_words_page", "sum_age", "amount_threegrams",
                  "amount_fourgrams"]
        return self.__get_features(ignore)


class Counter:
    """ The class that process and stores the wordtypes and phrasetypes features
    """

    def __init__(self):
        """ Initialise the class with default values
        """
        self.statement = 0  # Normal statement (terminates for example with . and !)
        self.question = 0  # Question (terminates with ?)

        self.mass = 0  # amount of mass (to check the amount of the filtered tokens)

        # POS TAGS (count them)
        self.dict = {"CC": 0, "CD": 0, "DT": 0, "EX": 0, "FW": 0, "IN": 0, "JJ": 0, "JJR": 0, "JJS": 0, "LS": 0,
                     "MD": 0, "NN": 0, "NNS": 0, "NNP": 0, "NNPS": 0, "PDT": 0, "POS": 0, "PRP": 0, "PRP$": 0, "RB": 0,
                     "RBR": 0, "RBS": 0, "RP": 0, "SYM": 0, "TO": 0, "UH": 0, "VB": 0, "VBD": 0, "VBG": 0, "VBN": 0,
                     "VBP": 0, "VBZ": 0, "WDT": 0, "WP": 0, "WP$": 0, "WRB": 0, "#": 0, "$": 0, ".": 0, ",": 0, ":": 0,
                     "(": 0, ")": 0, "``": 0, "''": 0, "NFP": 0, "HYPH": 0, "-LRB-": 0, "-RRB-": 0, "AFX": 0, "ADD": 0}
        self.universal_pos = {"ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CCONJ": 0, "DET": 0, "INTJ": 0, "NOUN": 0,
                              "NUM": 0, "PART": 0, "PRON": 0, "PROPN": 0, "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 0,
                              "X": 0}

        # POS TAG sentence part (nominal phrase etc.)
        self.adjp = 0  # Adjective phrase
        self.avg_adjp = 0  # Average amount of prepositional phrases per sentence
        self.advp = 0  # Adverb phrase
        self.avg_advp = 0  # Average amount of adverb phrases per sentence
        self.np = 0  # Noun phrase
        self.avg_np = 0  # Average amount of noun phrases per sentence
        self.pp = 0  # Prepositional phrase
        self.avg_pp = 0  # Average amount of prepositional phrases per sentence
        self.s = 0  # Simple declarative clause
        self.avg_s = 0  # Average amount of declarative clause per sentence
        self.frag = 0  # Fragment
        self.avg_frag = 0  # Average amount of fragments per sentence
        self.sbar = 0  # Subordinate clause
        self.avg_sbar = 0  # Average amount of subordinate clauses per sentence
        self.sbarq = 0  # Direct question introduced by wh-element
        self.avg_sbarq = 0  # Average amount of questions introduced by wh-element per sentence
        self.sinv = 0  # Declarative sentence with subject-aux inversion
        self.avg_sinv = 0  # Average amount of declarative sentences with subject-aux inversion per sentence
        self.sq = 0  # Questions without wh-element and yes/no questions
        self.avg_sq = 0  # Average amount of questions without wh-element and yes/no questions per sentence
        self.vp = 0  # Verb phrase
        self.avg_vp = 0  # Average amount of verb phrases per sentence
        self.whadvp = 0  # Wh-adverb phrase
        self.avg_whadvp = 0  # Average amount of wh-adverb phrases per sentence
        self.whnp = 0  # Wh-noun phrase
        self.avg_whnp = 0  # Average amount of wh-noun phrases per sentence
        self.whpp = 0  # Wh-prepositional phrase
        self.avg_whpp = 0  # Average amount of wh-prepositional phrases per sentence

        self.avg_phrases = 0  # Average amount of phrases per sentence

    def inc_statement(self):
        """ Increments the amount of statements
        """
        self.statement += 1

    def inc_question(self):
        """ Increments the amount of questions
        """
        self.question += 1

    def inc_mass(self):
        """ Increments the amount of mass
        """
        self.mass += 1

    def get_amount_adj(self):
        """ Returns the amount of adjectives
        :return: Amount of adjectives
        """
        return self.universal_pos["ADJ"]

    def get_amount_adposition(self):
        """ Returns the amount of adpositions
        :return: Amount of adpositions
        """
        return self.universal_pos["ADP"]

    def get_amount_noun(self):
        """ Returns the amount of nouns
        :return: Amount of nouns
        """
        return self.universal_pos["NOUN"] + self.universal_pos["PROPN"]

    def get_amount_pronoun(self):
        """ Returns the amount of pronouns
        :return: Amount of pronouns
        """
        return self.universal_pos["PRON"]

    def get_amount_verb(self):
        """ Returns the amount of verbs
        :return: Amount of verbs
        """
        return self.universal_pos["VERB"] + self.universal_pos["AUX"]

    def get_amount_main_verb(self):
        """ Returns the amount of main verbs
        :return: Amount of main verbs
        """
        return self.universal_pos["VERB"]

    def get_amount_auxiliary(self):
        """ Returns the amount of adjectives
        :return: Amount of adjectives
        """
        return self.universal_pos["AUX"]

    def get_amount_adverb(self):
        """ Returns the amount of adverbs
        :return: Amount of adverbs
        """
        return self.universal_pos["ADV"]

    def get_amount_coordinate_conj(self):
        """ Returns the amount of coordinate conjunctions
        :return: Amount of coordinate conjunctions
        """
        return self.universal_pos["CCONJ"]

    def get_amount_determiner(self):
        """ Returns the amount of determiners
        :return: Amount of determiners
        """
        return self.universal_pos["DET"]

    def get_amount_interjection(self):
        """ Returns the amount of interjections
        :return: Amount of interjections
        """
        return self.universal_pos["INTJ"]

    def get_amount_num(self):
        """ Returns the amount of numbers
        :return: Amount of numbers
        """
        return self.universal_pos["NUM"]

    def get_amount_particle(self):
        """ Returns the amount of particles
        :return: Amount of particles
        """
        return self.universal_pos["PART"]

    def get_amount_subord_conjuction(self):
        """ Returns the amount of subordinate conjunctions
        :return: Amount of subordinate conjunctions
        """
        return self.universal_pos["SCONJ"]

    def get_amount_foreign_word(self):
        """ Returns the amount of foreign words
        :return: Amount of foreign words
        """
        return self.dict["FW"]

    def get_amount_content_word(self):
        """ Returns the amount of content words
        :return: Amount of content words
        """
        amount_content_words = self.universal_pos["ADJ"] + self.universal_pos["ADV"] + self.universal_pos["INTJ"] + \
                               self.universal_pos["NOUN"] + self.universal_pos["PROPN"] + self.universal_pos["VERB"]
        return amount_content_words

    def get_amount_function_word(self):
        """ Returns the amount of function words
        :return: Amount of function words
        """
        amount_function_words = self.universal_pos["ADP"] + self.universal_pos["AUX"] + self.universal_pos["CCONJ"] + \
                                self.universal_pos["DET"] + self.universal_pos["NUM"] + self.universal_pos["PART"] + \
                                self.universal_pos["PRON"] + self.universal_pos["SCONJ"]
        return amount_function_words

    def get_amount_filtered(self):
        """ Calculates the number of all filtered tokens
        :return: Number of all filtered tokens
        """
        return self.universal_pos["PUNCT"] + self.universal_pos["SYM"] + self.universal_pos["X"] - self.dict["FW"] - \
               self.mass

    def store_pos(self, pos):
        """ Increment the specific pos tag (xpos)
        :param pos: specific pos tag (xpos)
        """
        if pos in self.dict:
            self.dict[pos] += 1
        else:
            print("Is not in dictionary: " + pos)  # For tests

    def store_upos(self, upos):
        """ Increment the specific pos tag (upos)
        :param pos: specific pos tag (upos)
        """
        if upos in self.universal_pos:
            self.universal_pos[upos] += 1
        else:
            print("Is not in dictionary: " + upos)  # For tests

    def get_phrases(self, cli, sentence):
        """  From a given sentence/line this function counts all types of phrases and stores them
        :param cli: the client of corenlp to parse the sentence to get all tokens of the grammatical structure.
        :param sentence: The sentence/line which has to be checked
        """
        annotated = cli.annotate(sentence)
        parsed = annotated['sentences'][0]['parse'].replace("\r\n", " ")

        self.adjp += parsed.count("(ADJP ")
        self.advp += parsed.count("(ADVP ")
        self.np += parsed.count("(NP ")
        self.pp += parsed.count("(PP ")
        self.s += parsed.count("(S ")
        self.frag += parsed.count("(FRAG ")
        self.sbar += parsed.count("(SBAR ")
        self.sbarq += parsed.count("(SBARQ ")
        self.sinv += parsed.count("(SINV ")
        self.sq += parsed.count("(SQ ")
        self.vp += parsed.count("(VP ")
        self.whadvp += parsed.count("(WHADVP ")
        self.whnp += parsed.count("(WHNP ")
        self.whpp += parsed.count("(WHPP ")

    def calculate_avg_sentence_parts(self, amount_sentences):
        """ Calculates the average of phrases and specific phrases per sentence/line
        :param amount_sentences: amount of sentences/lines
        """
        divisor = 1
        if amount_sentences:
            divisor = amount_sentences
        self.avg_adjp = self.adjp / divisor
        self.avg_advp = self.advp / divisor
        self.avg_np = self.np / divisor
        self.avg_pp = self.pp / divisor
        self.avg_s = self.s / divisor
        self.avg_frag = self.frag / divisor
        self.avg_sbar = self.sbar / divisor
        self.avg_sbarq = self.sbarq / divisor
        self.avg_sinv = self.sinv / divisor
        self.avg_sq = self.sq / divisor
        self.avg_vp = self.vp / divisor
        self.avg_whadvp = self.whadvp / divisor
        self.avg_whnp = self.whnp / divisor
        self.avg_whpp = self.whpp / divisor
        self.avg_phrases = self.get_amount_phrases() / divisor

    def get_amount_phrases(self):
        """ Calculates and returns the amount phrases
        :return: Amount of phrases
        """
        return self.adjp + self.advp + self.np + self.pp + self.s + self.frag + self.sbar + self.sbarq + self.sinv + \
               self.sq + self.vp + self.whadvp + self.whnp + self.whpp

    def get_features(self):
        """ Stores the values of all important attributes of this class (Counter) to an array and returns it.
        :return: Array with all the necessary attribute-values.
        """
        features = []
        amount_phrases = self.get_amount_phrases()
        amount_sentence_types = self.statement + self.question
        for element in self.__dict__.items():
            if element[0] not in ["dict", "universal_pos", "mass"]:
                features.append(round(element[1], 6))  # no ratio
                if element[0] in ["statement", "question"]:  # ratio
                    features.append(round(ratio(element[1], amount_sentence_types), 6))
                elif "avg" not in element[0]:
                    features.append(round(ratio(element[1], amount_phrases), 6))
        return features
