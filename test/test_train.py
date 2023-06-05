import numpy as np
import pytest
import src.pre_process
from src.train import train, data_transform
from src.evaluate import evaluation


@pytest.fixture()
def df():
    df = src.pre_process.load_data()
    yield df

def test_nondeterminism_robustness(df):
    corpus, dataset = src.pre_process.pre_process(df)
    X, y = data_transform(corpus, dataset)
    print("PRINT TEST")
    accs = []
    for seed in [1, 2, 3, 4, 5]:
        _, X_test, _, y_test, classifier = train(X, y, seed, seed)
        acc = evaluation(classifier, X_test, y_test)
        accs.append(acc)
    assert np.mean(accs) > 0.65
    assert np.var(accs) < 0.01
