# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from train.ml.data import process_data
from train.ml.model import inference
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


class Data(BaseModel):
    age: int = Field(..., example=28)
    workclass: str = Field(..., example="Private")
    fnlwgt: int = Field(..., example=338409)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=20)
    marital_status: str = Field(..., example="Married-civ-spouse")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="India")


@app.get("/")
async def default():
    return "FastAPI inference"


@app.post('/inference')
async def predict(data: Data):
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

    model, encoder, lb, metrics = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "model/model.pkl"))
    data_dict = data.__dict__
    keys = list(data_dict.keys())
    for feature in keys:
        modified_feature = feature.replace('_', '-')
        data_dict[modified_feature] = [data_dict[feature]]
        if modified_feature != feature:
            del data_dict[feature]

    df = pd.DataFrame(data_dict)
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False
    )
    y = inference(model, X)
    return {"output": lb.inverse_transform(y)[0]}
