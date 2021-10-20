# Script to train machine learning model.
import argparse
import logging

import dvc.api
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import set_config
from sklearn.compose import (TransformedTargetRegressor, make_column_selector,
                             make_column_transformer)
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

set_config(display='diagram')

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def create_slice(test, column_name, pipeline, encoder):
    uniques = test[column_name].unique()

    metrics_file = open(
        'starter/model/%s_slice_output.txt' %
        (column_name), "w")
    metrics_file.write(
        "Performance results for the slices of the feature %s" % (column_name))
    metrics_file.write("\n")

    for unique in uniques:
        slice = test[test[column_name] == unique]
        x_test, y_test = slice.drop(['salary'], axis=1), slice['salary']
        preds = pipeline.predict(x_test)
        precision, recall, fbeta = compute_model_metrics(
            encoder.transform(y_test), preds)
        metrics_file.write("Precision: %f" % (precision))
        metrics_file.write("\n")
        metrics_file.write("Recall: %f" % (recall))
        metrics_file.write("\n")
        metrics_file.write("F1: %f" % (fbeta))
        metrics_file.write("\n")
        metrics_file.write("*-*" * 5)
        metrics_file.write("\n")

    metrics_file.close()


def create_preprocessor():
    """
    Creates an instation of the sklearn preprocessor pipeline
    """

    # Selectors extract the column names that contains the specified dtype
    cat_selector = make_column_selector(dtype_include=object)
    num_selector = make_column_selector(dtype_include=np.number)

    # As our categorical column does not have to much different unique values we are going to use one_hot_encoder
    # We are going to use also a standarscaler for scaling and a imputer as we
    # have some NaNs in our badroom column
    categorical_processor = OneHotEncoder(handle_unknown="ignore")
    numerical_processor = make_pipeline(
        StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
    )

    # make column transformer to automaticaly do one_hot_encoder where there are categorical columns
    # and standarscaler where there are numerical columns
    preprocessor = make_column_transformer(
        (numerical_processor, num_selector), (categorical_processor, cat_selector))

    return preprocessor


def create_label_binarizer():
    label_binarizer = LabelBinarizer()
    return label_binarizer


def create_pipeline(model):
    """
    Creates a TransformedTargetRegressor pipeline with

    :param: modelcv: (Sklearn model with built-in cross-validation)

    :returns: A sklearn pipeline instantiation.
    """

    pipeline = TransformedTargetRegressor(
        regressor=model,
        func=None,
        inverse_func=None,
        check_inverse=False,
    )
    return pipeline


def main(args):
    # Add code to load in the data.
    with dvc.api.open(
        "starter/data/census_cleaned.csv",
            repo="https://github.com/gaceladri/nd0821-c3-starter-code") as fd:
        data = pd.read_csv(fd)

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.25, random_state=1)

    x_train, y_train = train.drop(['salary'], axis=1), train['salary']
    x_test, y_test = test.drop(['salary'], axis=1), test['salary']

    preprocessor = create_preprocessor()
    pipeline = create_pipeline(
        xgb.XGBClassifier(use_label_encoder=False))

    model = make_pipeline(
        preprocessor,
        pipeline
    )

    label_binarizer = create_label_binarizer()
    y_train_bin = label_binarizer.fit_transform(y_train)
    y_test_bin = label_binarizer.transform(y_test)

    model.fit(x_train, y_train_bin)
    pd.to_pickle(model, '../starter/model/pipeline.pkl')
    pd.to_pickle(label_binarizer, '../starter/model/label_binarizer.pkl')

    if args.slice_eval:
        create_slice(test, args.column_to_slice, model, label_binarizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic training of the model")

    parser.add_argument(
        "--slice_eval",
        type=bool,
        help="Wether or not we want to save a txt file with info about our metrics on a specific sliced column."
    )

    parser.add_argument(
        "--column_to_slice",
        type=str,
        help="The name of the column that is in the dataframe and we want to get the sliced metrics."
    )

    args = parser.parse_args()

    main(args)
