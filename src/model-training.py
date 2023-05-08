# Importing libraries
import numpy as np
import pandas as pd
import os

import re                           # re = Secret Labs' Regular Expression Engine
import nltk                         # NLTK = Natural Language Toolkit

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Variables
path_to_input_data = './data/input/'
path_to_output = './data/output/'
path_to_model = './data/trained_models/'
dataset_name = 'a1_RestaurantReviews_HistoricDump.tsv'
pkl_file_name = 'c1_BoW_Sentiment_Model'
classifier_name = 'c2_Classifier_Sentiment_Model'

path_to_dataset = path_to_input_data + dataset_name

# Importing dataset
dataset = pd.read_csv(path_to_dataset, delimiter = '\t', quoting = 3)
print(dataset.shape)


def pre_processing():
    """
    Data pre-processing part.
    :return: whatever is needed for other methods
    """
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus


def train():
    """
    Everything training related (to get to a model).
    :return:
    """
    # Dividing dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Model fitting (Naive Bayes)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, classifier


def predict(classifier, X_test):
    y_pred = classifier.predict(X_test)

    # Model performance
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    version_number = len(os.listdir(path_to_model))

    corpus = pre_processing()

    # Data transformation
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Saving BoW dictionary to later use in prediction
    bow_path = path_to_output + pkl_file_name + "_" + str(version_number) + '.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    X_train, X_test, y_train, y_test, classifier = train()

    # Exporting NB Classifier to later use in prediction
    joblib.dump(classifier, path_to_model + classifier_name + "_" + str(version_number))

    predict(classifier, X_test)

