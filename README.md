# HealthQuote-AI
HealthQuote AI Insurance is a machine learning project designed to predict insurance costs based on individual characteristics such as age, gender, BMI, number of children, smoking status, and region. By leveraging a Linear Regression model, the project aims to provide accurate estimates of insurance charges, helping users understand potential costs based on their profiles.

# Table of Contents
1.Installation
2.Usage
3.Project Structure
4.Data Description
5.Data Visualization
6.Data Preprocessing
7.Model Training
8.Model Evaluation
9.Saving and Loading the Model
10.Making Predictions

# Installation
Clone the Repository: Start by cloning the GitHub repository to your local machine.
Navigate to the Project Directory: Move into the project directory where the files are stored.
Install Dependencies: Install all the necessary Python libraries and packages as listed in the requirements.txt file.

# Usage
Loading and Inspecting Data
The project uses a dataset of insurance records, loaded into a DataFrame for analysis. This dataset contains features such as age, gender, BMI, number of children, smoking status, and region, along with the insurance charges. Initially, the data is inspected to understand its structure, types, and any potential issues like missing values.

# Data Visualization
To get a deeper understanding of the dataset, various visualizations are created. These include:
Histograms: To explore the distribution of continuous variables like age, BMI, and charges.
Count Plots: To analyze categorical variables such as gender, smoking status, number of children, and region.
These visualizations help in identifying patterns, trends, and outliers in the data.

# Data Preprocessing
Before training the model, the categorical variables (e.g., gender, smoker status, region) are converted into numerical values. This step is crucial as most machine learning algorithms, including Linear Regression, require numerical input. The preprocessing step also involves managing data types to ensure consistency across the dataset.

# Model Training
The preprocessed data is then split into training and testing sets. A Linear Regression model is trained on the training set, where it learns the relationship between the input features (age, gender, BMI, etc.) and the target variable (insurance charges). This model is the core of the project, enabling the prediction of insurance costs based on new data.

# Model Evaluation
After training, the model’s performance is evaluated on both the training and testing sets. The R-squared value, a common metric for regression models, is used to measure how well the model explains the variability of the target variable. This evaluation helps in understanding the accuracy and generalization of the model.

# Saving and Loading the Model
Once a satisfactory model is trained, it is saved to a file using joblib. This allows the model to be easily loaded and reused for future predictions without the need to retrain it.

# Making Predictions
With the model saved, predictions can be made on new data. By providing new input values (e.g., age, BMI, etc.), the model can estimate the expected insurance charges for an individual.

# Project Structure
Dataset (new.csv): Contains the data used for training and testing the model.
Model File (insurance_model.pkl): The trained Linear Regression model, saved for reuse.
Requirements (requirements.txt): Lists all necessary Python libraries and dependencies.
Documentation (README.md): Provides an overview and instructions for the project.

# Data Description
The dataset consists of 955 records, each with the following features:
Age: The age of the individual./n
Sex: Gender of the individual (encoded as 0 for male, 1 for female).
BMI: Body Mass Index, a measure of body fat based on height and weight.
Children: The number of children covered by the insurance.
Smoker: Smoking status (encoded as 0 for smoker, 1 for non-smoker).
Region: The geographical region of residence (encoded as numerical values).
Charges: The medical insurance costs billed.
