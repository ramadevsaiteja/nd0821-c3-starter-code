import pandas as pd
import pytest
import os


@pytest.fixture(scope='session')
def data():
    return pd.read_csv(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 '../data/clean_census.csv')))