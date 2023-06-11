# Importing libraries
import os
import pickle
import numpy as np

import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.evaluate import evaluation
from src.pre_process import pre_process

# Variables
PATH_TO_OUTPUT = './data/output/'
PATH_TO_MODEL = './data/trained_models/'
PKL_FILE_NAME = 'c1_BoW_Sentiment_Model'
CLASSIFIER_NAME = 'c2_Classifier_Sentiment_Model'


def train(features, labels, split_seed=0, model_seed=0):
    """
    Everything training related (to get to a model).
    :return:
    """
    # Dividing dataset into training and test set
    X_train, X_test, y_train, y_test \
        = train_test_split(features, labels, test_size=0.20, random_state=split_seed)

    # Model fitting (Naive Bayes)
    return X_test, y_test, GaussianNB().fit(X_train, y_train)


def data_transform(corpus, dataset, save_transform=False):
    # Data transformation
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    # Saving BoW dictionary to later use in prediction
    if save_transform:
        bow_path = PATH_TO_OUTPUT + PKL_FILE_NAME + "_" + str(version_number) + '.pkl'
        pickle.dump(cv, open(bow_path, "wb"))
    return X, y


def main():
    version_number = len(os.listdir(PATH_TO_MODEL))

    corpus, dataset = pre_process()

    X, y = data_transform(corpus, dataset, save_transform=True)

    X_test, y_test, classifier = train(X, y)

    # Exporting NB Classifier to later use in prediction
    joblib.dump(classifier, PATH_TO_MODEL + CLASSIFIER_NAME + "_" + str(version_number))

    evaluation(classifier, X_test, y_test)


if __name__ == '__main__':
    main()
