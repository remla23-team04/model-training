from sklearn.metrics import confusion_matrix, accuracy_score


def evaluation(classifier, X_test, y_test):
    """
    Evaluation
    :param classifier:
    :param X_test:
    :return:
    """
    y_pred = classifier.predict(X_test)

    # Model performance
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))