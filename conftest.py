import sys
sys.path.append('C:/Users/tmoore/PycharmProjects/nd0821-c3-starter-code/starter/ml')
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os
from data.data import process_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model import train_model, compute_model_metrics, inference
import pytest

@pytest.fixture
def X_train():
    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'census_cleaned.csv'))
    data = pd.read_csv(filename)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
        )
    X_train, _, _, _ = train_test_split(X, y, test_size=0.20)
    return X_train

@pytest.fixture
def y_train():
    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'census_cleaned.csv'))
    data = pd.read_csv(filename)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
        )
    _, _, y_train, _ = train_test_split(X, y, test_size=0.20)
    return y_train

@pytest.fixture
def y():
    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'census_cleaned.csv'))
    data = pd.read_csv(filename)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    _, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
        )
    return y

@pytest.fixture
def preds():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'census_cleaned.csv'))
    data = pd.read_csv(filename)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
        )
    preds = inference(model, X)

    return preds

@pytest.fixture
def X():
    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data', 'census_cleaned.csv'))
    data = pd.read_csv(filename)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
        )
    return X

@pytest.fixture
def model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    return model
