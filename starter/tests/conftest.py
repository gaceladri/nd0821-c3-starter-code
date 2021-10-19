import dvc
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def data():
    with dvc.api.open(
        "starter/data/census_cleaned.csv",
            repo="https://github.com/gaceladri/nd0821-c3-starter-code") as fd:
        data = pd.read_csv(fd)

    if data is None:
        pytest.fail("The data could not be downloaded from dvc")

    return data
