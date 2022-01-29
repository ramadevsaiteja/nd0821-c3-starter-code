import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from train.ml.data import process_data
from train.ml.model import inference
import joblib


def test_process_data(data):
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
    X_train, y_train, encoder, lb = process_data(
        data.head(10), categorical_features=cat_features, label="salary", training=True
    )
    assert len(X_train) == len(y_train)


def test_for_null_values(data):
    assert data.isnull().any().all() == False


def test_inference_process(data):
    model, encoder, lb, metrics = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                           "model/model.pkl"))

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
    X_test, y_test, encoder, lb = process_data(
        data.head(10), categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    preds = inference(model=model, X=X_test)
    assert len(y_test) == len(preds)
