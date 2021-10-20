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
    'https://coursera-income-pred.herokuapp.com/prediction/', data=json.dumps(data))

print(response.status_code)
print(response.json())