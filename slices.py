import os
import sys
sys.path.append('./ml')
import pickle

from model import slices
from data import process_data
import pandas as pd

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/OneHot.pkl", "rb") as f:
    oneHot = pickle.load(f)

with open("model/LabelBinarizer.pkl", "rb") as f:
    lb = pickle.load(f)

df = pd.read_csv("data/census_cleaned.csv")

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

slices(model, "education", df, oneHot, lb, cat_features)