# HealthQuote-AI

HealthQuote AI Insurance is a machine learning project designed to predict insurance costs based on individual characteristics such as age, gender, BMI, number of children, smoking status, and region. By leveraging a Linear Regression model, this project aims to provide accurate estimates of insurance charges, helping users understand potential costs based on their profiles.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Data Description](#data-description)
5. [Data Visualization](#data-visualization)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Saving and Loading the Model](#saving-and-loading-the-model)
10. [Making Predictions](#making-predictions)
11. [Deploying with FastAPI](#deploying-with-fastapi)

## Installation

1. **Clone the Repository**: Start by cloning the GitHub repository to your local machine.
2. **Navigate to the Project Directory**: Move into the project directory where the files are stored.
3. **Install Dependencies**: Install all the necessary Python libraries and packages as listed in the `requirements.txt` file.

```bash
git clone https://github.com/yourusername/healthquote-ai.git
cd healthquote-ai
pip install -r requirements.txt
```

## Usage

### Loading and Inspecting Data

The project uses a dataset of insurance records, loaded into a DataFrame for analysis. This dataset contains features such as age, gender, BMI, number of children, smoking status, region, and insurance charges. Initially, the data is inspected to understand its structure, data types, and check for any missing values.

## Data Visualization

To gain a deeper understanding of the dataset, various visualizations are created, including:

- **Histograms**: To explore the distribution of continuous variables like age, BMI, and charges.
- **Count Plots**: To analyze categorical variables such as gender, smoking status, number of children, and region.

These visualizations help in identifying patterns, trends, and outliers in the data.

## Data Preprocessing

Before training the model, categorical variables (e.g., gender, smoker status, region) are converted into numerical values. This step is essential as most machine learning algorithms, including Linear Regression, require numerical input. Data types are also managed to ensure consistency across the dataset.

## Model Training

The preprocessed data is split into training and testing sets. A Linear Regression model is trained on the training set, where it learns the relationship between the input features (age, gender, BMI, etc.) and the target variable (insurance charges). This model is the core of the project, enabling the prediction of insurance costs based on new data.

## Model Evaluation

After training, the model’s performance is evaluated on both the training and testing sets. The R-squared value, a common metric for regression models, is used to measure how well the model explains the variability of the target variable. This evaluation helps assess the accuracy and generalization of the model.

## Saving and Loading the Model

Once a satisfactory model is trained, it is saved to a file using joblib. This allows the model to be easily loaded and reused for future predictions without needing to retrain it.

## Making Predictions

With the model saved, predictions can be made on new data. By providing new input values (e.g., age, BMI, etc.), the model can estimate the expected insurance charges for an individual.

## Deploying with FastAPI

To make the insurance prediction model accessible via an API, FastAPI is used. Below is a quick guide on how to deploy the model:

### 1. FastAPI Setup

Ensure FastAPI is installed:

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

### 2. Create the API

Create a new Python file (e.g., `main.py`) and include the following code:

```python
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
```

### 3. Run the API

You can start the API using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, and you can send POST requests to `/predict` to get insurance cost predictions.

## Project Structure

- **Dataset (`new.csv`)**: Contains the data used for training and testing the model.
- **Model File (`insurance_model.pkl`)**: The trained Linear Regression model, saved for reuse.
- **API Code (`main.py`)**: FastAPI implementation to serve predictions.
- **Requirements (`requirements.txt`)**: Lists all necessary Python libraries and dependencies.
- **Documentation (`README.md`)**: Provides an overview and instructions for the project.

## Data Description

The dataset consists of 955 records, each with the following features:

- **Age**: The age of the individual.
- **Sex**: Gender of the individual (encoded as 0 for male, 1 for female).
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **Children**: The number of children covered by the insurance.
- **Smoker**: Smoking status (encoded as 0 for smoker, 1 for non-smoker).
- **Region**: The geographical region of residence (encoded as numerical values).
- **Charges**: The medical insurance costs billed.
