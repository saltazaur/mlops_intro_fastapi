from joblib import dump, load

class IrisModel:
    def __init__(self, model = "iris_model.joblib"):
        self.model = model
        self.logreg = load(self.model)
        
    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        dimensions = [sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]
    
        prediction = self.logreg.predict([dimensions])[0]
    
        # Make prediction readible for a human
        iris_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        human_readible_prediction = iris_dict[prediction]        
        
        return human_readible_prediction