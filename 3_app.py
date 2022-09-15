from fastapi import FastAPI
from mlmodel import IrisModel

app = FastAPI(title = "🌸 Iris Species", description = "an app supporting gardeners in data-driven decisions")
model = IrisModel()


@app.get("/model_introduction")
def intro():
    return {"model_version": "0.0.1",
            "model_name": "Iris Species",
            "model_author": "Accenture",
            "app_author": "Best Courses"}

@app.post("/prediction")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    species = model.predict(sepal_length, sepal_width, petal_length, petal_width)
    return species


