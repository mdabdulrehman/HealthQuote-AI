from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('insurance_model.pkl')  # Ensure the file name matches the saved model

# Initialize the FastAPI app
app = FastAPI()

# Define the input data model
class InsuranceData(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int

@app.post('/predict')
def predict_insurance(data: InsuranceData):
    # Convert input data to numpy array
    input_data = np.array([[data.age, data.sex, data.bmi, data.children, data.smoker, data.region]])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the prediction
    return {'insurance_cost': prediction[0]}