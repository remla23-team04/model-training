# Importing libraries
import numpy as np
import os

import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Variables
path_to_output = './data/output/'
path_to_model = './data/trained_models/'
pkl_file_name = 'c1_BoW_Sentiment_Model'
classifier_name = 'c2_Classifier_Sentiment_Model'

from evaluate import evaluation
from pre_process import pre_process


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


if __name__ == '__main__':
    version_number = len(os.listdir(path_to_model))

    corpus, dataset = pre_process()

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

    evaluation(classifier, X_test, y_test)
