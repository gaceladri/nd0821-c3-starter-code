import dvc
import pandas as pd
import pytest


@pytest.fixture(scope='sesion')
def data():
    data = dvc.api.read(
        "/starter/data/census_cleaned.csv",
        repo="https://github.com/gaceladri/nd0821-c3-starter-code",
        mode="r"
    )

    if data is None:
        pytest.fail("The data could not be downloaded from dvc")

    df = pd.read_csv(data)

    return df
