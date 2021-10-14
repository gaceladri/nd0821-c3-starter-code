import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_client_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello world!"

def test_column_names():
    request_data = {
        "age":25,
        'workclass':"Never-married",
        'fnlgt': 71345,
        'education':"Some-college",
        'education-num':10,
        'marital-status':"Never-married",
        'occupation':"Machine-op-inspct",
        'relationship':"Unmarried",
        'race':"White",
        'sex':"Male",
        'capital-gain':0,
        'capital-loss':0,
        'hours___per-week':35,
        'native-country':"India",
    }
    r = client.post(
        "http://127.0.0.1:8000/prediction/",
        data=json.dumps(request_data))
    assert r.status_code != 200 

def test_column_types():
    request_data = {
        "age":"25",
        'workclass':"Never-married",
        'fnlgt': 71345,
        'education':"Some-college",
        'education-num':10,
        'marital-status':"Never-married",
        'occupation':"Machine-op-inspct",
        'relationship':"Unmarried",
        'race':"White",
        'sex':"Male",
        'capital-gain':0,
        'capital-loss':0,
        'hours___per-week':35,
        'native-country':"India",
    }
    r = client.post(
        "http://127.0.0.1:8000/prediction/",
        data=json.dumps(request_data))
    assert r.status_code != 200 