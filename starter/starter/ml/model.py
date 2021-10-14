from sklearn.metrics import fbeta_score, precision_score, recall_score


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
