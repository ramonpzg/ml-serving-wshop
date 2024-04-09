
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class InputData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


def load_model():
    return joblib.load("my_model.joblib")


model = load_model()

@app.post("/predict")
def predict(data: InputData):
    # Convert input data to a 2D array
    features = [[
        data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
        data.magnesium, data.total_phenols, data.flavanoids,
        data.nonflavanoid_phenols, data.proanthocyanins, data.color_intensity,
        data.hue, data.od280_od315_of_diluted_wines, data.proline
    ]]
    
    # Make predictions using the loaded model
    prediction = model.predict(features)
    
    # Return the predicted class
    return {"class": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
