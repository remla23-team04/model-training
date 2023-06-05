import pytest
# import pandas

@pytest.fixture()
def df():
    # df = pandas.read()
    df = []
    yield df

def test_something(df):
    assert True
