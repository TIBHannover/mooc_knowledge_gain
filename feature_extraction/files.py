import os
import sys
import csv
import fitz
import processor
import client
import embedding
import re
import scipy
import numpy
import torch
from operator import itemgetter
from SortedCollection import SortedCollection


def load_stop_words(path):
    """ Loads stopwords of the stopwords.txt file to use them later
    :param path: the path where the file exist
    :return: Array of stopwords
    """
    try:
        stops = open(path, 'r')
        lines = stops.readlines()
        stopwords = []
        for line in lines:
            stopwords.append(line.replace("\n", ""))
        return stopwords
    except IOError:
        print("Couldn't load stopwords.txt")
        sys.exit(1)


def load_embd_model(name):
    """ Loads embedding model. the program will stop if the name does not exist
    :param name: name of model
    :return: specific model
    """
    try:
        return embedding.get_model(name)
    except IOError:
        print("Couldn't load sentence-embedding model")
        sys.exit(1)


def get_embedding_array(name, length):
    """ Creates an array of strings where every string represents a predictore of the embedding.
    :param name: name that is used for the strings
    :param length: length of embedding
    :return: array with the names
    """
    arr = []
    i = 1
    while i <= length:
        arr.append(name + "_" + str(i))  # enumerate predicores (gives them a number)
        i += 1
    return arr


def store_embeddings_to_features(features, embeddings_sli, embeddings_tra):
    """ Takes the average embeddings of slides and transcripts and includes them into array of features for a specific
    video
    :param features: features of videos
    :param embeddings_sli: average embedding of slides
    :param embeddings_tra: average embedding of transcript
    :return: features and average embeddings as one array
    """
    embd_pos = 0
    for i in range(len(features)):
        for y in range(len(features[i])):  # iterates through videos
            new_list = features[i][y]
            knowledge_gain = new_list[-1]
            knowledge_gain_value = new_list[-2]
            new_list = new_list[:-2]  # Removes the knowledge gain values to put them later to the end of the list
            # Stores every predictor of the embeddings as one feature
            for z in range(len(embeddings_sli[embd_pos])):
                new_list.append(embeddings_sli[embd_pos][z])
            for z in range(len(embeddings_tra[embd_pos])):
                new_list.append(embeddings_tra[embd_pos][z])
            embd_pos += 1
            new_list.append(knowledge_gain_value)
            new_list.append(knowledge_gain)
            features[i][y] = new_list
    return features


def load_csv(path):
    """ Loads a specific csv-file. The program will be stopped if the file does not exist
    :param path: the path of the file
    :return: the loaded file
    """
    try:
        csv_file = open(path, 'r', newline='', encoding="ISO-8859-1")
        return csv_file
    except IOError:
        print("Couldn't load  csv-file")
        sys.exit(1)


def get_z_values(test):
    """ Loads the knowledge gain of the test values (csv-file) and calculates them into z-values to get a nominal
    distribution. This distribution is used to classify the knowledge gain into one of three knowledge gain classes.
    :param test:
    :return:
    """
    test.seek(1)
    reader = csv.reader(test, delimiter=',')
    values = []
    first = True  # ignores header
    for row in reader:
        if first:
            first = False
        else:
            values.append(float(row[len(row) - 1]))  # last entry contains the knowledge gain
    print("Convert Knowledge Gain values into nominal ones:")
    print("Mean:" + str(scipy.mean(values)))
    print("Standard Deviation:" + str(numpy.std(values)))
    converted = scipy.stats.zscore(values)
    print("Converted.")
    return converted


def process_test(test, name, z_scores, slides_result, transcript_result, embedding_features):
    """ Generates all rows of a video for the result csv-file. The method checks how many persons watched the video.
    Every person represents a row and contains their knowledge gain
    :param test: csv-file with knowledge gain of persons
    :param name: name of the video
    :param z_scores: z_scores of all knowledge gains
    :param slides_result: features of the slides
    :param transcript_result: features of the transcript
    :param embedding_features: features of the embeddings for the video
    :return: rows that represent all information of a video
    """
    name = name.replace("video", "")  # test file has not video in the name of videos
    rows = []
    found = False  # shows if correct video values were already found
    pos = -1  # shouldn't check header so begin with -1 (-1 will automatically goes to 0)
    test.seek(1)  # go to start of file again (so every video can be processed)
    reader = csv.reader(test, delimiter=',')
    for row in reader:  # Iterates through all rows
        if name in row:  # checks if row is about the specific video
            # generates row with features
            person_id = [row[1]]
            # get visual features
            visual_features = row[2: len(row) - 3]
            knowledge_gain = [row[len(row) - 1]]
            # converts knowledge gain to nominal value
            if -0.5 <= z_scores[pos] <= 0.5:
                knowledge_gain_level = ["Moderate"]
            elif z_scores[pos] < -0.5:
                knowledge_gain_level = ["Low"]
            else:
                knowledge_gain_level = ["High"]
            rows.append([name] + person_id + slides_result + transcript_result + embedding_features
                        + visual_features + knowledge_gain + knowledge_gain_level)
        elif found:
            break
        pos += 1
    return rows


def create_csv(rows):
    """ Stores all the extracted features into a csv file
    :param rows: rows that contains the information of the videos (every row has information about one video)
    """
    features = open('./Features/all_features.csv', 'w', newline='')
    text_features = open('./Features/text_features.csv', 'w', newline='')
    multimedia_features = open('./Features/multimedia_features.csv', 'w', newline='')
    avg_slide_embedding = open('./Features/slide_embedding.csv', 'w', newline='')
    avg_transcript_embedding = open('./Features/transcript_embedding.csv', 'w', newline='')
    text_features_writer = csv.writer(text_features, delimiter=',')
    multimedia_features_writer = csv.writer(multimedia_features, delimiter=',')
    features_writer = csv.writer(features, delimiter=',')
    slide_writer = csv.writer(avg_slide_embedding, delimiter=',')
    transcript_writer = csv.writer(avg_transcript_embedding, delimiter=',')
    slides_features = ["amount_tok_sli",
                       "amount_uni_tok_sli", "ratio_uni_tok_sli", "amount_uni_lemma_sli", "ratio_uni_lemma_sli",
                       "sum_tok_len_sli", "min_tok_len_sli",
                       "avg_tok_len_sli", "max_tok_len_sli", "avg_freq_tok_sli",
                       "avg_trigram_sli", "avg_tetragram_sli", "min_line_len", "avg_line_len",
                       "max_line_len", "min_line_chars", "avg_line_chars", "max_line_chars",
                       "amount_syl_sli", "amount_one_syl_sli", "amount_two_syl_sli",
                       "amount_psyl_sli", "amount_hard_sli", "avg_syl_sli", "ratio_one_syl_sli",
                       "ratio_two_syl_sli", "ratio_psyl_sli", "ratio_hard_sli",
                       "min_age_sli", "avg_age_sli", "max_age_sli",
                       "amount_slides", "sum_lines", "min_lines", "avg_lines", "max_lines",
                       "min_words_slide", "avg_words_slide", "max_words_slide", "flesch_ease_sli", "flesch_kin_sli",
                       "gunning_fog_sli", "smog_sli", "ari_sli", "coleman_sli",
                       "read_time_sli", "amount_adj_sli", "avg_adj_sli",
                       "ratio_adj_sli", "amount_adpos_sli", "avg_adpos_sli", "ratio_adpos_sli",
                       "amount_noun_sli", "avg_noun_sli",
                       "ratio_noun_sli", "amount_pronoun_sli", "avg_pronoun_sli",
                       "ratio_pronoun_sli", "ratio_pronoun_noun_sli", "amount_verb_sli", "avg_verb_sli",
                       "ratio_verb_sli",
                       "amount_main_verb_sli", "avg_main_verb_sli", "ratio_main_verb_sli", "amount_aux_sli",
                       "avg_aux_sli", "ratio_aux_sli",
                       "amount_adverb_sli", "avg_adverb_sli", "ratio_adverb_sli", "amount_coord_conj_sli",
                       "avg_coord_conj_sli", "ratio_coord_conj_sli", "amount_determiner_sli",
                       "avg_determiner_sli", "ratio_determiner_sli",
                       "amount_interj_sli", "avg_interj_sli", "ratio_interj_sli", "amount_num_sli",
                       "avg_num_sli", "ratio_num_sli",
                       "amount_particle_sli", "avg_particle_sli", "ratio_particle_sli", "amount_subord_conj_sli",
                       "avg_subord_conj_sli", "ratio_subord_conj_sli", "amount_foreign_sli",
                       "avg_foreign_sli", "ratio_foreign_sli",
                       "amount_content_word_sli", "avg_content_word_sli", "ratio_content_word_sli",
                       "amount_function_word_sli", "avg_function_word_sli", "ratio_function_word_sli",
                       "amount_filtered_sli", "avg_filtered_sli", "ratio_filtered_sli",
                       "amount_statement_sli", "ratio_statement_sli",
                       "amount_question_sli", "ratio_question_sli", "ADJP_sli", "ratio_ADJP_sli", "avg_ADJP_sli",
                       "ADVP_sli",
                       "ratio_ADVP_sli", "avg_ADVP_sli",
                       "NP_sli", "ratio_NP_sli", "avg_NP_sli", "PP_sli", "ratio_PP_sli", "avg_PP_sli",
                       "S_sli", "ratio_S_sli", "avg_S_sli", "FRAG_sli", "ratio_FRAG_sli", "avg_FRAG_sli",
                       "SBAR_sli", "ratio_SBAR_sli", "avg_SBAR_sli", "SBARQ_sli", "ratio_SBARQ_sli", "avg_SBARQ_sli",
                       "SINV_sli", "ratio_SINV_sli", "avg_SINV_sli", "SQ_sli", "ratio_SQ_sli", "avg_SQ_sli",
                       "VP_sli", "ratio_VP_sli", "avg_VP_sli", "WHADVP_sli", "ratio_WHADVP_sli", "avg_WHADVP_sli",
                       "WHNP_sli", "ratio_WHNP_sli", "avg_WHNP_sli", "WHPP_sli", "ratio_WHPP_sli", "avg_WHPP_sli",
                       "avg_phrases_sli",
                       "sim_pres_sli", "ratio_sim_pres_sli", "pres_prog_sli", "ratio_pres_prog_sli",
                       "pres_perf_sli", "ratio_pres_perf_sli", "pres_perf_prog_sli", "ratio_pres_perf_prog_sli",
                       "sim_pas_sli", "ratio_sim_pas_sli", "pas_prog_sli", "ratio_pas_prog_sli",
                       "pas_perf_sli", "ratio_pas_perf_sli", "pas_perf_prog_sli", "ratio_pas_perf_prog_sli",
                       "will_sli", "ratio_will_sli", "fu_prog_sli", "ratio_fu_prog_sli", "fu_perf_sli",
                       "ratio_fu_perf_sli",
                       "fu_perf_prog_sli", "ratio_fu_perf_prog_sli", "cond_sim_sli", "ratio_cond_sim_sli",
                       "cond_prog_sli", "ratio_cond_prog_sli", "cond_perf_sli", "ratio_cond_perf_sli",
                       "cond_perf_prog_sli", "ratio_cond_perf_prog_sli",
                       "gerund_sli", "ratio_gerund_sli", "perf_part_sli", "ratio_perf_part_sli",
                       "inf_sli", "ratio_inf_sli", "perf_inf_sli", "ratio_perf_inf_sli",
                       "active_sli", "ratio_active_sli", "passive_sli", "ratio_passive_sli"]
    transcript_features = ["amount_sentences", "amount_tok_tra",
                           "amount_uni_tok_tra", "ratio_uni_tok_tra", "amount_uni_lemma_tra", "ratio_uni_lemma_tra",
                           "sum_tok_len_tra", "min_tok_len_tra",
                           "avg_tok_len_tra", "max_tok_len_tra", "avg_freq_tok_tra", "avg_trigram_tra",
                           "avg_tetragram_tra", "min_sen_len", "avg_sen_len",
                           "max_sen_len", "min_sen_chars", "avg_sen_chars", "max_sen_chars",
                           "amount_syl_tra", "amount_one_syl_tra", "amount_two_syl_tra",
                           "amount_psyl_tra", "amount_hard_tra", "avg_syl_tra", "ratio_one_syl_tra",
                           "ratio_two_syl_tra", "ratio_psyl_tra", "ratio_hard_tra",
                           "min_age_tra", "avg_age_tra", "max_age_tra", "flesch_ease_tra", "flesch_kin_tra",
                           "gunning_fog_tra", "smog_tra", "ari_tra", "coleman_tra", "read_time_tra",
                           "speak_time",
                           "speak_difference", "amount_subtitles", "amount_adj_tra", "avg_adj_tra", "ratio_adj_tra",
                           "amount_adpos_tra", "avg_adpos_tra", "ratio_adpos_tra", "amount_noun_tra",
                           "avg_noun_tra", "ratio_noun_tra",
                           "amount_pronoun_tra", "avg_pronoun_tra", "ratio_pronoun_tra", "ratio_pronoun_noun_tra",
                           "amount_verb_tra", "avg_verb_tra", "ratio_verb_tra",
                           "amount_main_verb_tra", "avg_main_verb_tra", "ratio_main_verb_tra", "amount_aux_tra",
                           "avg_aux_tra",
                           "ratio_aux_tra", "amount_adverb_tra", "avg_adverb_tra", "ratio_adverb_tra",
                           "amount_coord_conj_tra", "avg_coord_conj_tra", "ratio_coord_conj_tra",
                           "amount_determiner_tra", "avg_determiner_tra",
                           "ratio_determiner_tra", "amount_interj_tra", "avg_interj_tra",
                           "ratio_interj_tra",
                           "amount_num_tra", "avg_num_tra", "ratio_num_tra", "amount_particle_tra",
                           "avg_particle_tra", "ratio_particle_tra",
                           "amount_subord_conj_tra", "avg_subord_conj_tra", "ratio_subord_conj_tra",
                           "amount_foreign_tra", "avg_foreign_tra", "ratio_foreign_tra",
                           "amount_content_word_tra", "avg_content_word_tra", "ratio_content_word_tra",
                           "amount_function_word_tra", "avg_function_word_tra", "ratio_function_word_tra",
                           "amount_filtered_tra", "avg_filtered_tra", "ratio_filtered_tra",
                           "amount_statement_tra", "ratio_statement_tra", "amount_question_tra", "ratio_question_tra",
                           "ADJP_tra", "ratio_ADJP_tra", "avg_ADJP_tra", "ADVP_tra", "ratio_ADVP_tra", "avg_ADVP_tra",
                           "NP_tra", "ratio_NP_tra", "avg_NP_tra", "PP_tra", "ratio_PP_tra", "avg_PP_tra",
                           "S_tra", "ratio_S_tra", "avg_S_tra", "FRAG_tra", "ratio_FRAG_tra", "avg_FRAG_tra",
                           "SBAR_tra", "ratio_SBAR_tra", "avg_SBAR_tra", "SBARQ_tra", "ratio_SBARQ_tra",
                           "avg_SBARQ_tra", "SINV_tra", "ratio_SINV_tra", "avg_SINV_tra",
                           "SQ_tra", "ratio_SQ_tra", "avg_SQ_tra", "VP_tra", "ratio_VP_tra", "avg_VP_tra",
                           "WHADVP_tra", "ratio_WHADVP_tra", "avg_WHADVP_tra", "WHNP_tra", "ratio_WHNP_tra",
                           "avg_WHNP_tra", "WHPP_tra", "ratio_WHPP_tra", "avg_WHPP_tra", "avg_phrases_tra",
                           "sim_pres_tra", "ratio_sim_pres_tra", "pres_prog_tra", "ratio_pres_prog_tra",
                           "pres_perf_tra", "ratio_pres_perf_tra", "pres_perf_prog_tra", "ratio_pres_perf_prog_tra",
                           "sim_pas_tra", "ratio_sim_pas_tra", "pas_prog_tra", "ratio_pas_prog_tra",
                           "pas_perf_tra", "ratio_pas_perf_tra", "pas_perf_prog_tra", "ratio_pas_perf_prog_tra",
                           "will_tra", "ratio_will_tra", "fu_prog_tra", "ratio_fu_prog_tra", "fu_perf_tra",
                           "ratio_fu_perf_tra",
                           "fu_perf_prog_tra", "ratio_fu_perf_prog_tra", "cond_sim_tra", "ratio_cond_sim_tra",
                           "cond_prog_tra", "ratio_cond_prog_tra", "cond_perf_tra", "ratio_cond_perf_tra",
                           "cond_perf_prog_tra", "ratio_cond_perf_prog_tra", "gerund_tra", "ratio_gerund_tra",
                           "perf_part_tra", "ratio_perf_part_tra", "inf_tra", "ratio_inf_tra",
                           "perf_inf_tra", "ratio_perf_inf_tra", "active_tra",
                           "ratio_active_tra", "passive_tra", "ratio_passive_tra"]
    embedding_features = ["similarity_sli", "similarity_tra", "diff_similarity", "similarity_vectors"]
    visual_features = ["Clear_Language", "Vocal_Diversity", "Filler_Words", "Speed_of_Presentation",
                       "Coverage_of_the_Content",
                       "Level_of_Detail", "Highlight", "Summary", "Text_Design", "Image_Design", "Formula_Design",
                       "Table_Design",
                       "Structure", "Entry_Level", "Overall_Rating", "loudness_avg", "mod_loudness_avg",
                       "rms_energy_avg",
                       "f0_avg", "jitter_avg", "delta_jitter_avg", "shimmer_avg", "harmonicity_avg", "log_HNR_avg",
                       "PVQ_avg", "speech_rate", "articulation_rate", "average_syllable_duration", "txt_ratio_avg",
                       "txt_ratio_var", "img_ratio_avg", "img_ratio_var", "highlight", "level_of_detailing_avg",
                       "level_of_detailing_var",
                       "coverage_of_slide_content_avg", "coverage_of_slide_content_var"]
    avg_embedding_slides = get_embedding_array("avg_embd_slides_dim", 16)
    avg_embedding_transcript = get_embedding_array("avg_embd_transcript_dim", 16)
    features_writer.writerow(["Video_ID"] + ["Person_ID"] + slides_features + transcript_features + embedding_features
                             + visual_features + ["Knowledge_Gain", "Knowledge_Gain_Level"])
    text_features_writer.writerow(
        ["Video_ID"] + ["Person_ID"] + slides_features + transcript_features + embedding_features
        + ["Knowledge_Gain", "Knowledge_Gain_Level"])
    multimedia_features_writer.writerow(["Video_ID"] + ["Person_ID"] + visual_features + ["Knowledge_Gain",
                                                                                          "Knowledge_Gain_Level"])
    slide_writer.writerow(
        ["Video_ID"] + ["Person_ID"] + avg_embedding_slides + ["Knowledge_Gain", "Knowledge_Gain_Level"])
    transcript_writer.writerow(
        ["Video_ID"] + ["Person_ID"] + avg_embedding_transcript + ["Knowledge_Gain", "Knowledge_Gain_Level"])

    # write values inside csv-files
    for line in rows:
        for row in line:
            i = 0
            stop = 2 + len(slides_features) + len(transcript_features) + len(embedding_features)
            feature_values = []
            text_feature_values = []
            multimedia_feature_values = []
            slides_embd = []
            transcript_embd = []
            # store video_id and user_id
            while i < 2:
                feature_values.append(row[i])
                text_feature_values.append(row[i])
                multimedia_feature_values.append(row[i])
                slides_embd.append(row[i])
                transcript_embd.append(row[i])
                i += 1
            # store text-features
            while i < stop:
                feature_values.append(row[i])
                text_feature_values.append(row[i])
                i += 1
            # store multimedia-features
            stop += len(visual_features)
            while i < stop:
                feature_values.append(row[i])
                multimedia_feature_values.append(row[i])
                i += 1
            stop += len(avg_embedding_slides)
            # store slides-embedding
            while i < stop:
                slides_embd.append(row[i])
                i += 1
            stop += len(avg_embedding_transcript)
            # store transcript-embedding
            while i < stop:
                transcript_embd.append(row[i])
                i += 1

            # store knowledge-gain
            feature_values.append(row[-2])
            feature_values.append(row[-1])
            text_feature_values.append(row[-2])
            text_feature_values.append(row[-1])
            multimedia_feature_values.append(row[-2])
            multimedia_feature_values.append(row[-1])
            slides_embd.append(row[-2])
            slides_embd.append(row[-1])
            transcript_embd.append(row[-2])
            transcript_embd.append(row[-1])

            #store to files
            features_writer.writerow(feature_values)
            text_features_writer.writerow(text_feature_values)
            multimedia_features_writer.writerow(multimedia_feature_values)
            slide_writer.writerow(slides_embd)
            transcript_writer.writerow(transcript_embd)

    # close files
    features.close()
    text_features.close()
    multimedia_features.close()
    avg_slide_embedding.close()
    avg_transcript_embedding.close()


def remove_files(files):
    """ Deletes files. This method is used to deletes old files when the user starts a new calculation
    :param files: Array of old files
    """
    for file in files:
        os.remove(file)


def load_files(path):
    """ Loads files of an path
    :param path: the path where the files exist
    :return: array of strings of the path of files
    """
    path_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            path = root + '/' + file
            # replace \ to / to get for windows/linux/mac the same representation
            path_files.append(path.replace('\\', '/'))
    return path_files


def process_files(files, sta, cli):
    """ Process all files of the videos to generate the features. The pdf files and the srt files must have the same
    name to know that they belong to the same video. Otherwise the program can't recognize it and stops the
    calculation.
    :param files: tuple of files for the videos
    :param sta:  stanza object to get word specific features
    :param cli: client to generate sentence trees
    """
    rows = []
    embeddings_sli = []  # contains average embeddings for slides
    embeddings_tra = []  # contains average embeddings for transcripts
    properties = load_csv('./wordlists/freq_syll_words.csv')
    age = load_csv('./wordlists/AoA_51715_words.csv')
    test = load_csv('./Data/Test/test.csv')
    stopwords = load_stop_words('./wordlists/stopwords.txt')
    z_scores = get_z_values(test)
    model = load_embd_model('roberta-large-nli-stsb-mean-tokens')
    for slides, transcript in files:
        try:
            s = open(slides, 'rb')
            t = open(transcript, 'rb')
            s_name = os.path.basename(s.name[:-4])
            t_name = os.path.basename(t.name[:-4])
            if s_name != t_name:  # Check if slides and transcript have the same name
                print("Names of slides and transcript must be the same.")
                client.stop_client(cli)
                properties.close()
                test.close()
                sys.exit(1)
            # get features and stores them
            features, embd_sli, embd_tra = (process_video(s, t, s_name, sta, cli, properties, age,
                                                          test, z_scores, model, stopwords))
            rows.append(features)
            for i in range(len(features)):
                embeddings_sli.append(embd_sli)
                embeddings_tra.append(embd_tra)
            s.close()
            t.close()

            # clean gpu cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except IOError:
            print("Can't open slides or transcript for a video. Video will be ignored.")
            pass
    properties.close()
    test.close()
    client.stop_client(cli)
    # reduce dimension of embeddings to have a better representation
    embeddings_sli = embedding.reduce_dimension(embeddings_sli)
    embeddings_tra = embedding.reduce_dimension(embeddings_tra)
    create_csv(store_embeddings_to_features(rows, embeddings_sli, embeddings_tra))
    print("Stored features as csv-files in ./Features")


def process_video(slides, transcript, name, sta, cli, properties, age, test, z_scores, model, stopwords):
    """ Process a specific video. The files for slides and transcript are used to get the features about this video.
    Also all important objects are passed to realize the calculation
    :param slides: files for slides
    :param transcript: files for transcript
    :param name: name of the video
    :param sta: stanza to calculate word specific features
    :param cli: client to calculate sentence trees
    :param properties:  csv-table with amount of syllables and frequency for words
    :param age: csv-table that includes the age of acquisition for words
    :param test: csv-file with the knowledge gains
    :param z_scores: values of the knowledge gains as z-score to calculate the nominal classes
    :param model: the sentence embedding model to calculate the embeddings
    :param stopwords: stopwords that has to be filtered for the frequency
    :return: rows of features for the specific video
    """
    print("Process slide: " + name + ".pdf")
    slides_result, slides_lines = process_slides(slides, sta, cli, properties, age, stopwords)
    print("Finished process of slide: " + name + ".pdf")
    print("Process transcript: " + name + ".srt")
    transcript_result, transcript_sentences = process_transcript(transcript, sta, cli, properties, age, stopwords)
    embd_features, embd_sli, embd_tra = embedding.process_video_embeddings(slides_lines, transcript_sentences, model)
    print("Finished process of  transcript: " + name + ".srt")
    return process_test(test, name, z_scores, slides_result, transcript_result, embd_features), embd_sli, embd_tra


def process_slides(slides, sta, cli, properties, age, stopwords):
    """ Calculates the feature for specific slides of a video
    :param slides: the slides of a video
    :param sta: stanza to calculate word specific features
    :param cli: client to calculate sentence trees
    :param properties:  csv-table with amount of syllables and frequency for words
    :param age: csv-table that includes the age of acquisition for words
    :param stopwords: stopwords that has to be filtered for the frequency
    :return: features of the slides
    """
    calculator = processor.Calculator()
    features, sentences = calculator.process_lines(slides.readlines(), cli, sta, processor.Counter(), properties, age,
                                                   stopwords)
    return features, sentences


def process_transcript(transcript, sta, cli, properties, age, stopwords):
    """ Calculates the feature for specific transcript of a video
    :param transcript: the transcript of a video
    :param sta: stanza to calculate word specific features
    :param cli: client to calculate sentence trees
    :param properties:  csv-table with amount of syllables and frequency for words
    :param age: csv-table that includes the age of acquisition for words
    :param stopwords: stopwords that has to be filtered for the frequency
    :return: features of the slides
    """
    calculator = processor.Calculator()
    features, sentences = calculator.process_sentences(transcript.read(), cli, sta, processor.Counter(), properties,
                                                       age, stopwords)
    return features, sentences


def write_pdf(location, pages):
    """ Stores the textual aspects of a pdf into a txt-file.
    :param location: location to store it
    :param pages: array of pages that contains an array of lines for every page
    """
    f = open(location, "w", encoding="utf-8")
    number = 0
    for page in pages:
        # Every page gets the beginning line "Starting next page:" to clarify that a new page started
        f.write("Starting next page:" + str(number) + "\n")
        for line in page:
            """
            Replaces special characters with whitespace or nothing to get a better formatted string/line.
            Also replace -\xad‐ to - because so it is the same as the text in the pdf. 
            Add \n to the string to see it later as one line.
            """
            f.write(line[4].replace("\xa0", "").replace("\r", "").replace("\t ", " ").replace("\t", " ")
                    .replace("-\xad‐", "‐") + "\n")
        number += 1
    f.close()


def write_embeddings(location, embeddings):
    """ Method to store embeddings as txt file (not used but can be useful)
    :param location: the location to store them
    :param embeddings: the embeddings to store
    """
    f = open(location, "w")
    for embedding in embeddings:
        """the first entry of an embedding represents the normal sentence. This sentence ist stored in " " to identify
        what the embedding represents. Every predictor is separated with a whitespace """
        f.write('"' + embedding[0] + '"')
        for emb in embedding[1]:
            f.write(" " + str(emb.item()))
        f.write('\n')
    f.close()


def positions(lines):
    """ Stores every line in a new array and checks if the order is correct or the line should be placed earlier.
    :param lines: lines to check
    :return: array of lines with new order
    """
    result = []
    for line in lines:
        if len(result) > 1:  # array has more then 1 element
            # check backwards if previous lines are more on the right side then the current one
            for i in range(len(result) - 1, -1, -1):
                current = result[i]
                diff = current[3] - line[3]
                """ Checks if previous line has similar y coordinate and a higher x coordinate. If this is true the 
                order will be replaced to the position where this is no more true
                """
                if (0 >= diff >= -5) and (current[0] > line[0]):
                    if i == 0:  # reached end. Replaces it to the beginning
                        result.insert(0, line)
                        break
                    continue
                else:
                    if i < (len(result) - 1):  # Replaces it to the correct position
                        result.insert(i + 1, line)
                        break
                    result.append(line)
                    break
        elif len(result) == 1:  # 1 element in array
            current = result[0]
            diff = current[3] - line[3]
            if (0 >= diff >= -5) and (current[0] > line[0]):
                result.insert(0, line)
            else:
                result.append(line)
        else:  # empty array
            result.append(line)
    return result


def contains_letters(text):
    """ Check if text is bigger then 1 and check if text contains letters. This is a helpfunction to check if a line is
    useful or useless. A character or only special characters / numbers have no meaning.
    :param text: text to check the criteria
    :return: Returns True if text is longer than 1 and contains a letter otherwise False
    """
    return len(text) > 1 and re.search("[a-zA-Z]", text)


def process_pdf(file):
    """" Converts a PDF file to a txt file to get the text.
    :param file: PDF file which should be converted
    """
    doc = fitz.open(file)
    # print(file)
    i = 0
    pages = []
    for page in doc:
        block_page = page.getTextWords()
        sorted_blocks = SortedCollection(key=itemgetter(5, 6, 7))
        for block in block_page:
            sorted_blocks.insert(block)
        lines = merge_line(sorted_blocks)
        sorted_lines = SortedCollection(key=itemgetter(3, 0))
        for line in lines:
            # print(line)
            # Checks if line is bigger then 1 and has a letter.
            if contains_letters(line[4]):
                sorted_lines.insert(line)
        # print()
        sorted_lines = positions(sorted_lines)
        i += 1
        pages.append(sorted_lines)
    write_pdf("./Data/Slides-Processed/" + os.path.basename(file).replace(".pdf", ".txt"), pages)


def change_values(word1, word2, text, string):
    """ Merges the text of two words.
    :param word1: first word
    :param word2: second word
    :param text: the text that represents the words
    :param string:
    :return: changed first word
    """
    word1[text] = word1[text] + string + word2[text]
    return word1


def merge_line(words):
    """ Merge words-objects (lines) to one object (line) if they have the same block_no, a different line_no and a
    a maximum difference for y0 and y1 of 2. The object that starts first is the beginning of the merged object.
    :param words: words-object (lines) to manipulate
    :return: merged objects list
    """
    merged = []
    for word in words:
        if not merged:
            merged.append(word)
        else:
            prev = list(merged[len(merged) - 1])
            # get values of objects
            block_no1, line_no1, word_no1 = prev[5], prev[6], prev[7]
            block_no2, line_no2, word_no2 = word[5], word[6], word[7]
            # checks if both words-objects are in the same block and line. If it is true merge the prev with the next.
            if block_no1 == block_no2 and line_no1 == line_no2:
                prev = change_values(prev, word, 4, " ")
                merged[len(merged) - 1] = prev
            # Checks if objects have same block_no and different line_no to look deeper for merging
            elif block_no1 == block_no2 and line_no1 != line_no2:
                #  Checks if y0 and y1 coordinates are similar between the objects (merge criteria)
                if (abs(prev[1] - word[1]) <= 2) and (abs(prev[3] - word[3]) <= 2):
                    diff = prev[0] - word[0]
                    """ checks if the x0 coordinate of the previous one is higher. If it is higher the word will be 
                    append to it. Otherwise the word will be prepend to it"""
                    if diff > 0:
                        word = change_values(list(word), prev, 4, "\t")
                        merged[len(merged) - 1] = word
                    else:
                        prev = change_values(prev, word, 4, "\t")
                        merged[len(merged) - 1] = prev
                else:  # no merge
                    merged.append(word)
            else:  # no merge
                merged.append(word)
    return merged
