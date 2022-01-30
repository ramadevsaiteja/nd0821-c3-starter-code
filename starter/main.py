# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from train.ml.data import process_data
from train.ml.model import inference

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

    model, encoder, lb, metrics = joblib.load('./model/model.pkl')
    dict = data.__dict__
    keys = list(dict.keys())
    for feature in keys:
        modified_feature = feature.replace('_', '-')
        dict[modified_feature] = [dict[feature]]
        if modified_feature != feature:
            del dict[feature]

    df = pd.DataFrame(dict)
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False
    )
    y = inference(model, X)
    return {"output": lb.inverse_transform(y)[0]}
