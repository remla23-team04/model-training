import numpy as np
import pytest
import src.pre_process
from src.train import train, data_transform
from src.evaluate import evaluation
from time import time
import random
import memory_profiler

# To run without suppressing stdout:
# `pytest -rP`

@pytest.fixture()
def df():
    df = src.pre_process.load_data()
    yield df

def test_label_distribution(df):
    """
    Test that the distribution of labels is sufficiently even in the training data.
    """
    num_pos = df[df["Liked"] == 1].shape[0]
    num_neg = df[df["Liked"] == 0].shape[0]
    assert np.abs(num_pos / num_neg - 1) < 0.25

def test_preprocessing(df):
    """
    Test that the preprocessing function behaves as expected. 
    """
    corpus, dataset = src.pre_process.pre_process(df)
    assert dataset.shape[0] == len(corpus)
    for sentence in corpus:
        assert "." not in sentence
        assert sentence == sentence.lower()

def test_data_slice(df):
    """
    Test that reviews containing exclamation marks have a similar accuracy to reviews in general
    Reviews with exclamation marks are an important data slice because they mean the reviewer was enthusiastic (either positively or negatively)
    """
    df_exc = df[df['Review'].str.contains("!")].reset_index(drop=True)
    df_non_exc = df[~df['Review'].str.contains("!")].reset_index(drop=True)
    corpus, dataset = src.pre_process.pre_process(df_non_exc)
    corpus_exc, _ = src.pre_process.pre_process(df_exc)
    # Transform main data set
    X, y, cv = data_transform(corpus, dataset)
    # Apply transform to exclamation mark set
    X_exc = cv.transform(corpus_exc).toarray()
    y_exc = df_exc.iloc[:, -1].values
    
    seed = 1
    X_test, y_test, classifier = train(X, y, seed, seed)
    main_acc = evaluation(classifier, X_test, y_test)
    exc_acc = evaluation(classifier, X_exc, y_exc)
    assert np.abs(main_acc - exc_acc) < 0.2

def test_nondeterminism_robustness(df):
    """
    Test that different trainings of the model do not lead to big differences in accuracy
    """
    corpus, dataset = src.pre_process.pre_process(df)
    X, y, _ = data_transform(corpus, dataset)
    accs = []
    for seed in [1, 2, 3, 4, 5]:
        X_test, y_test, classifier = train(X, y, seed, seed)
        acc = evaluation(classifier, X_test, y_test)
        accs.append(acc)
    assert np.mean(accs) > 0.65
    assert np.var(accs) < 0.01

def test_inference_performance(df):
    """
    Test that inference maintains certain execution speed and max memory usage requirements. 
    """
    corpus, dataset = src.pre_process.pre_process(df)
    X, y, _ = data_transform(corpus, dataset)
    seed = 1
    X_test, y_test, classifier = train(X, y, seed, seed)
    start = time()
    acc = evaluation(classifier, X_test, y_test)
    time_taken = time() - start
    assert time_taken < 0.1
    mem_usage = memory_profiler.memory_usage((evaluation, [classifier, X_test, y_test]))
    assert max(mem_usage) < 300

def extract_synonyms(phrase):
    synonyms = []
    #  antonyms = []
    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            synonyms.append(l.name())
        #    if l.antonyms():
        #         antonyms.append(l.antonyms()[0].name())
    return synonyms

def test_mutamorphic_synonym(df):
    """
    Mutamorphic test that ensures that replacing a sentiment-associated word with a synonym does not change performance greatly
    """
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    
    corpus, dataset = src.pre_process.pre_process(df)
    new_corpus = []
    for i in range(len(corpus)):
        # Get data instance
        words = corpus[i].split(" ")
        # 20% of words are selected along with their indices
        selected_words = random.sample(list(enumerate(words)), len(words) // 5)
        for j in range(len(selected_words)):
            idx, sel_word = selected_words[j]
            # Get synonyms for each word
            syns = extract_synonyms(sel_word)
            # If there are synonyms, pick one randomly
            syn = random.choice(syns) if syns else sel_word
            # Overwrite word with synonym at saved index
            words[idx] = syn
        # Add data instance to new dataset
        new_corpus.append(" ".join(words))
    # Train model on both datasets and verify performance similarity
    X, y, _ = data_transform(corpus, dataset)
    Xn, yn, _ = data_transform(new_corpus, dataset)
    accs = []
    seed = 1
    for X, y in [(X, y), (Xn, yn)]:
        X_test, y_test, classifier = train(X, y, seed, seed)
        acc = evaluation(classifier, X_test, y_test)
        accs.append(acc)
    assert len(accs) == 2
    assert np.abs(accs[0] - accs[1]) <= 0.1
