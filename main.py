# Put the code for your API here.
import os
import sys
sys.path.append('./ml')

from fastapi import FastAPI
from pydantic import BaseModel, Field
from model import inference, process_data
import pickle
import pandas as pd

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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

with open("model/model.pkl", "rb") as f:
    tmodel = pickle.load(f)

with open("model/OneHot.pkl", "rb") as f:
    oneHot = pickle.load(f)

with open("model/LabelBinarizer.pkl", "rb") as f:
    lb = pickle.load(f)

app = FastAPI()

class Census(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True

@app.get("/")
async def welcome_message():
    return {"greeting": "Welcome to my Udacity Machine Leanrings Dev Ops Engineering project"}

@app.post('/predict')
async def predict(person: Census):

    df = pd.DataFrame([person.dict()])
    df.rename(lambda x: x.replace("_", "-"), axis="columns", inplace=True)

    processed_df, _, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False, encoder=oneHot, lb=lb)

    prediction = inference(tmodel, processed_df)

    prediction = lb.inverse_transform(prediction)[0]

    return {"prediction": prediction}
