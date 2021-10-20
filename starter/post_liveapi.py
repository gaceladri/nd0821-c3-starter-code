import requests
import json

data = {
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
    'hours-per-week': 35,
    'native-country': "India",
}

response = requests.post(
    'http://127.0.0.1:8000/prediction/', data=json.dumps(data))
# https://coursera-income-pred.herokuapp.com/

print(response.status_code)
print(response.json())