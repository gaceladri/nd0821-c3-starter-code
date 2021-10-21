import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_client_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': 'Hello world!'}


def test_column_names():
    request_data = {
        "age": 25,
        'workclass': "Never-married",
        'fnlgt': 71345,
        'education': "Some-college",
        'education-num': 10,
        'marital-status': "Never-married",
        'occupation': "Machine-op-inspct",
        'relationship': "Unmarried",
        'race': "White",
        'sex': "Male",
        'capital-gain': 0,
        'capital-loss': 0,
        'hours___per-week': 35,
        'native-country': "India",
    }
    r = client.post(
        "http://127.0.0.1:8000/prediction/",
        data=json.dumps(request_data))
    assert r.status_code != 200


def test_column_types():
    request_data = {
        "age": "25",
        'workclass': "Never-married",
        'fnlgt': 71345,
        'education': "Some-college",
        'education-num': 10,
        'marital-status': "Never-married",
        'occupation': "Machine-op-inspct",
        'relationship': "Unmarried",
        'race': "White",
        'sex': "Male",
        'capital-gain': 0,
        'capital-loss': 0,
        'hours___per-week': 35,
        'native-country': "India",
    }
    r = client.post(
        "http://127.0.0.1:8000/prediction/",
        data=json.dumps(request_data))
    assert r.status_code != 200


def test_inference_above():
    request_data = {'age': 50,
                    'workclass': 'Self-emp-not-inc',
                    'fnlgt': 83311,
                    'education': 'Bachelors',
                    'education-num': 13,
                    'marital-status': 'Married-civ-spouse',
                    'occupation': 'Exec-managerial',
                    'relationship': 'Husband',
                    'race': 'White',
                    'sex': 'Male',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'hours-per-week': 13,
                    'native-country': 'United-States'}

    r = client.post(
        "http://127.0.0.1:8000/prediction/",
        data=json.dumps(request_data))

    assert r.status_code == 200
    assert r.json() == {"prediction": '<=50K'}


def test_inference_below():
    request_data = {'age': 52,
                    'workclass': 'Self-emp-not-inc',
                    'fnlgt': 209642,
                    'education': 'HS-grad',
                    'education-num': 9,
                    'marital-status': 'Married-civ-spouse',
                    'occupation': 'Exec-managerial',
                    'relationship': 'Husband',
                    'race': 'White',
                    'sex': 'Male',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'hours-per-week': 45,
                    'native-country': 'United-States', }

    r = client.post(
        "http://127.0.0.1:8000/prediction/",
        data=json.dumps(request_data))

    assert r.status_code == 200
    assert r.json() == {"prediction": '>50K'}
