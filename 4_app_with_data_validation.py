from fastapi import FastAPI
from pydantic import BaseModel, validator
from mlmodel import IrisModel

app = FastAPI(title = "🌸 Iris Species", description = "an app supporting gardeners in data-driven decisions")
model = IrisModel()


# A Pydantic model
class IrisDimensions(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    @validator("sepal_length")
    def check_sepal_length(cls, v):
        if v > 7.9:
            raise ValueError('The model was not trained for this range. Do you use a correct unit? cm?')
        if v < 4.3:
            raise ValueError('The model was not trained for this range. Do you use a correct unit? cm?')
        return v


@app.get("/model_introduction")
def intro():
    return {"model_version": "0.0.1",
            "model_name": "Iris Species",
            "model_author": "Accenture",
            "app_author": "Best Courses"}

@app.post("/prediction")
def predict(iris: IrisDimensions):
    species = model.predict(iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width)
    return species


