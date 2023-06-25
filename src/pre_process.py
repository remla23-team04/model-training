import re                           # re = Secret Labs' Regular Expression Engine
import nltk                         # NLTK = Natural Language Toolkit
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def load_data(path_to_input_data='./data/input/', dataset_name='a1_RestaurantReviews_HistoricDump.tsv'):
    """
    Loading the data from a set location
    :param path_to_input_data: folder starting from root directory
    :param dataset_name: file name (appended to path_to_input_data)
    :return: the loaded dataset
    """
    path_to_dataset = path_to_input_data + dataset_name
    # Importing dataset
    dataset = pd.read_csv(path_to_dataset, delimiter = '\t', quoting = 3)
    # print("dataset shape: ", dataset.shape)
    return dataset


def process_a_review(review, all_stopwords, ps):
    """
    Processing a review (should be the same as the processing of the prediction endpoint from model-service).
    :param review: the actual content - input
    :param all_stopwords: what words not to consider
    :param ps: used for stemming words
    :return: the processed review
    """
    # Substitute anything that is not a-zA-Z with a space
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return ' '.join(review)


def pre_process(dataset=load_data()):
    """
    Data pre-processing part.
    :return: whatever is needed for other methods
    """
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    for i in range(0, dataset.shape[0]):
        review = process_a_review(dataset['Review'][i], all_stopwords, ps)
        corpus.append(review)

    return corpus, dataset


if __name__ == "__main__":
    pre_process()
