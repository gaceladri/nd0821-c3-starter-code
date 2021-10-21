import sklearn
import xgboost

from starter.train_model import create_label_binarizer, create_preprocessor
from starter.train_model import create_pipeline

def test_uniques(data):
    expected_uniques = [
        'Bachelors',
        'HS-grad',
        '11th',
        'Masters',
        '9th',
        'Some-college',
        'Assoc-acdm',
        'Assoc-voc',
        '7th-8th',
        'Doctorate',
        'Prof-school',
        '5th-6th',
        '10th',
        '1st-4th',
        'Preschool',
        '12th']

    these_uniques = data["education"].unique()

    assert list(expected_uniques) == list(these_uniques)


def test_column_names(data):
    expected_names = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary']

    these_columns = data.columns.values
    
    assert list(these_columns) == list(expected_names)


def test_label_binarizer():
    lb = create_label_binarizer()
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer


def test_pipeline():
    pipeline = create_pipeline(xgboost.XGBClassifier(use_label_encoder=False))
    assert type(pipeline) == sklearn.compose.TransformedTargetRegressor

def test_preprocessor():
    preprocessor = create_preprocessor()
    assert type(preprocessor) == sklearn.compose.make_column_transformer