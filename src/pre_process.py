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
    print("dataset shape: ", dataset.shape)
    return dataset


def pre_process():
    """
    Data pre-processing part.
    :return: whatever is needed for other methods
    """
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    dataset = load_data()
    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus, dataset


if __name__ == "__main__":
    pre_process()