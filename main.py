from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Simple model train karo — ghar ka size se price predict karo
# (fake data — concept samjhne ke liye)
X = np.array([[500], [800], [1000], [1500], [2000]])
y = np.array([50000, 80000, 100000, 150000, 200000])

model = LinearRegression()
model.fit(X,y)

class HouseInput(BaseModel):
    size_sqft: float
    
@app.get("/")
def home():
    return {"status":"ML API chal rahi hai"}


@app.post("/predict")
def predict(house:HouseInput):
    prediction= model.predict([[house.size_sqft]])[0]
    return {
        "size_sqft": house.size_sqft,
        "predicted_price": round(prediction, 2),
        "currency": "USD"
    }
