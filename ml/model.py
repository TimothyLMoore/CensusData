import os
import sys

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=19)
    model.fit(X_train, y_train)

    return model



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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def slices(model, cat, X, encoder, lb, cat_features):
    """ Computes performance on model slices

    Inputs
    ------
    model : ???
        Trained machine learning model.
    cat : str
        category to be sliced
    X : np.array
        Data used for prediction.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, for processing data.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, for processing data.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])

    Returns
    -------
    No returns
    """
    with open("slice_output.txt", "a") as f:
        print(cat, file=f)
        for cls in X[cat].unique():
            X_temp = X[X[cat] == cls]
            X_temp, y_temp, encoder, lb = process_data(
                X_temp, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb=lb
                )
            preds = inference(model, X_temp)
            precision, recall, fbeta = compute_model_metrics(y_temp, preds)
            print("#################################", file=f)
            print(cls, file=f)
            print('Precision:', precision, file=f)
            print('Recall:', recall, file=f)
            print('F-Beta Score:', fbeta, file=f)
            print("#################################", file=f)

    return

print(__file__)
