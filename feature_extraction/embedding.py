from sentence_transformers import SentenceTransformer, util
from numpy import add
from torch import Tensor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_model(name):
    """ Loads the SentenceTransformer for the specific model. If the model is not stored it will be downloaded
    :param name: the name of the model
    :return: specific SentenceTransformer
    """
    # best performance has roberta-large-nli-stsb-mean-tokens
    return SentenceTransformer(name)


def get_embeddings(sentences, model):
    """ Creates the sentence embeddings from the sentences
    :param sentences: the sentences as array to calculate the embeddings from
    :param model: the SentenceTransformer model
    :return: Array of embeddings as tensor-object
    """
    embeddings = []
    for embedding in model.encode(sentences, convert_to_tensor=True):
        embeddings.append(embedding)
    return embeddings


def calculate_divisor(n):
    """ Calculates how many tuples of sentence will be compared. This information is used as divisor to calculate
    average values
    :param n: the amount of sentences
    :return: the amount of comparisons
    """
    divisor = 1
    n = (n - 1)
    while n > 1:  # do not add 1 to the divisor because it starts with 1
        divisor += n
        n = (n - 1)
    return divisor


def get_similarity(embeddings):
    """ This function calculates every similarity between two sentences and takes the average one
    :param embeddings: array of embeddings
    :return: average similarity
    """
    length = len(embeddings)
    if length == 0:  # when array is empty there is no similarity
        return 0.0
    elif length == 1:
        return 1.0
    else:
        i = 0
        similarity = 0
        while i < (length - 1):  # iterates through the array of embeddings
            z = i + 1
            other_embeddings = []
            while z < length:  # iterates through all following embeddings to form a pair with the current one
                other_embeddings.append(embeddings[z])
                score = util.pytorch_cos_sim(embeddings[i], embeddings[z])[0]  # consinus sim to show similarity
                similarity += score.item()
                z += 1
            i += 1
        return round(similarity / calculate_divisor(length), 2)  # rounding for better representation


def calculate_avg_embedding(embeddings):
    """ Calculates the average sentence embedding/vector of all sentence embeddings
    :param embeddings: Array of embeddings to calculate the average embedding
    :return: Average embedding (array of all coordinates)
    """
    avg = [0] * 1024  # default vector
    if not len(embeddings):  # if length is 0 return default to avoid division with zero
        return avg
    for emb in embeddings:
        avg = add(avg, emb.cpu().numpy())  # adds complete embedding to vector (numpy because embedding was an tensor object)
    # divides every predictor with the amount of embeddings to get the average number
    return [value / len(embeddings) for value in avg]


def reduce_dimension(embeddings):
    """ Reduce the dimension of embeddings so that they can be used in classification. A high dimension of 1024 would
    lead to a bad representation of the correlations. Other features would be drowned out by their small number
    :param embeddings: the embeddings with high dimension
    :return: reduced dimension embeddings
    """
    n = 16  # used for experiments
    print("Reduce Embeddings to dimension " + str(n))
    pca = PCA(n_components=n)
    x = StandardScaler().fit_transform(embeddings)
    values = pca.fit_transform(x)
    information = 0  # default amount of information
    for value in pca.explained_variance_ratio_:
        information += value
    print("Reduced embeddings. Embeddings contain about " + str(round(information, 4) * 100) + " % information")
    result = []
    for embd in values.tolist():
        # rounds values to avoid wrong representation of floating number
        result.append([round(value, 6) for value in embd])
    return result


def process_video_embeddings(slides, transcript, model):
    """ Calculates the embeddings for the slides and the transcript. After that the features are going to be calculated
    :param slides: An array with all lines of the slides (Array of Strings)
    :param transcript: An array with all sentences of the transcript (Array of Strings)
    :return: The calculated features
    """
    embeddings_slides = get_embeddings(slides, model)
    embeddings_transcript = get_embeddings(transcript, model)
    similarity_slides = round(get_similarity(embeddings_slides), 6)
    similarity_transcript = round(get_similarity(embeddings_transcript), 6)
    diff_similarity = round(abs(similarity_slides - similarity_transcript), 6)
    avg_slides = calculate_avg_embedding(embeddings_slides)
    avg_slides = [round(value, 6) for value in avg_slides]
    avg_transcript = calculate_avg_embedding(embeddings_transcript)
    avg_transcript = [round(value, 6) for value in avg_transcript]
    avg_vectors = [Tensor(avg_slides), Tensor(avg_transcript)]
    similarity_vector = round(get_similarity(avg_vectors), 6)
    features = [similarity_slides, similarity_transcript, diff_similarity, similarity_vector], avg_slides, \
               avg_transcript
    return features
