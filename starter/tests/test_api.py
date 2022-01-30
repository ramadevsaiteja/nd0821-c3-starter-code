from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "FastAPI inference"


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200


def test_post():
    data = {
        "age": 28,
        "workclass": "Private",
        "fnlwgt": 338409,
        "education": "Bachelors",
        "education_num": 20,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "India"
    }
    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert "output" in r.json()
    assert r.json()["output"] == "<=50K"


def test_post_malformed():
    data = {
        "tmp": 1
    }
    r = client.post('/inference', json=data)
    assert r.status_code != 200
