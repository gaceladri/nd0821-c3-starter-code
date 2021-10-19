# Put the code for your API here.

import os

import pandas as pd
from fastapi import Depends, FastAPI
from fastapi.security.oauth2 import OAuth2PasswordBearer
from pydantic import BaseModel, Field

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# To allow heroku to pull data from DVC
# Source: https://ankane.org/dvc-on-heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("Pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model = pd.read_pickle("./model/pipeline.pkl")


@app.get("/")
async def root():
    return {"message": "Hello world!"}


class Input(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Never-married")
    fnlgt: int = Field(..., example=74136)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=12)
    marital_status: str = Field(...,
                                alias="marital-status", example="Divorced")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., alias="capital-gain", example=1)
    capital_loss: int = Field(..., alias="capital-loss", example=1)
    hours_per_week: int = Field(..., alias="hours-per-week", example=35)
    native_country: str = Field(..., alias="native-country",
                                example="United-States")


class Output(BaseModel):
    prediction: str


@app.post("/prediction/", response_model=Output, status_code=200)
async def get_predictions(input: Input):
    input_dataframe = pd.Dataframe(input, index=[0])
    predictions = model.predict(input_dataframe)

    return {"prediction": predictions}
