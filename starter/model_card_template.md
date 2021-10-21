# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was made for the udacity course of Machine Learning DevOps Engineer Nanodegree Program. Currently the model is an XGBoost Classifier embedded into an SKLearn pipeline, with a label binarizer to encode the target feature.

This model was developed as a Proof Of Concept to apply good DevOps practices and deploy a model with DVC (Data Version Contro), FastAPI and Heroku for easy deployment. 

The model was developed 21th of October of 2021.

## Intended Use

The intended use of this model is to predict the salary income based on publicly available Census Bureau data.

## Training Data

The acquisition of the data is based on publicly available Census Bureau data. The model was trained with the 75% of the data, using a train-test set split.

## Evaluation Data

The acquisition of the data is based on publicly available Census Bureau data. The model was evaluated with the 25% of the data, using a train-test set split.

## Metrics


- precision: The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of true positives and ``fp`` the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.


The best value is 1 and the worst value is 0.

- recall: The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of true positives and ``fn`` the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

The best value is 1 and the worst value is 0.

- fbeta: The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of recall in the combined score. ``beta < 1`` lends more weight to precision, while ``beta > 1`` favors recall (``beta -> 0`` considers only precision, ``beta -> +inf`` only recall).

## Ethical Considerations

The data is publicly available, so the model can be used to disclose to wich degree a person's salary is associated with his or her age, education, location, etc. The model should be assessed with other data and models to address a more considerate conclusion.

## Caveats and Recommendations

The model is used for experimental use cases. Do not use it for real-world problems.