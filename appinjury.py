import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
# Load the dataset
file_path = 'injury1.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Feature Engineering: Creating a derived feature
data['Total_Training_Load'] = data['Match_Load'] + data['Training_Load']

# 1. Handle missing values and encoding
numeric_features = ['Age', 'Match_Load', 'Training_Load', 'Fatigue_Level', 'BMI', 'Total_Training_Load']
categorical_features = ['Role', 'Previous_Injury', 'Environmental_Factor', 'Diet_Quality']

# Preprocessor for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessor for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer that handles both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# 2. Define target (Injury_Risk) and features
X = data.drop(columns=['Player_ID', 'Injury_Risk'])  # Features
y = data['Injury_Risk']  # Target variable

# Convert the target variable into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Fit the preprocessor on the entire dataset (this speeds up prediction)
X_preprocessed = preprocessor.fit_transform(X)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_preprocessed, y_encoded)

# 3. Define injury types and solutions based on predictions
injury_info = {
    'High': {'type': 'Severe injuries (e.g., fractures, ligament tears)', 'solution': 'Immediate medical attention, rest, rehabilitation, and gradual return to play.'},
    'Medium': {'type': 'Moderate injuries (e.g., strains, sprains)', 'solution': 'Ice, rest, compression, elevation (RICE), and gradual recovery exercises.'},
    'Low': {'type': 'Minor injuries (e.g., bruises, minor sprains)', 'solution': 'Rest and self-care, may require over-the-counter pain relief.'}
}

# 4. Define prediction function
def predict_injury_risk(age, match_load, training_load, fatigue_level, bmi, role, previous_injury, environmental_factor, diet_quality):
    # Create input data as a dictionary
    input_data = {
        'Age': age,
        'Match_Load': match_load,
        'Training_Load': training_load,
        'Fatigue_Level': fatigue_level,
        'BMI': bmi,
        'Role': role,
        'Previous_Injury': previous_injury,
        'Environmental_Factor': environmental_factor,
        'Diet_Quality': diet_quality
    }

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Create Total_Training_Load feature in the input data
    input_df['Total_Training_Load'] = input_df['Match_Load'] + input_df['Training_Load']

    # Preprocess the input data
    input_preprocessed = preprocessor.transform(input_df)

    # Predict the injury risk
    prediction_encoded = model.predict(input_preprocessed)
    prediction_proba = model.predict_proba(input_preprocessed)[0]  # Get the prediction probabilities
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    # Get injury type and solution
    injury_type = injury_info[prediction]['type']
    solution = injury_info[prediction]['solution']

    # Get the risk probability for the predicted class
    risk_percentage = prediction_proba[prediction_encoded][0] * 100

    return prediction, risk_percentage, injury_type, solution, prediction_proba


# Streamlit UI Setup
st.title("Cricket Player Injury Risk Predictor")
st.write("Enter player details below to predict the injury risk and get suggested solutions.")

# Apply CSS styles
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Input sliders and dropdowns for user inputs
age = st.slider("Age (16-40)", 16, 40)
match_load = st.slider("Match Load (0-10 matches/week)", 0, 10)
training_load = st.slider("Training Load (0-10)", 0, 10)
fatigue_level = st.slider("Fatigue Level (1-10)", 1, 10)
bmi = st.slider("BMI (18.5-35)", 18.5, 35.0)

role = st.selectbox("Role", options=["Batsman", "Bowler", "All-Rounder"])
previous_injury = st.selectbox("Previous Injury", options=["Yes", "No"])
environmental_factor = st.selectbox("Environmental Factor", options=["Hot", "Cold", "Dry"])
diet_quality = st.selectbox("Diet Quality", options=["Poor", "Fair", "Good"])

# Prediction button
if st.button("Predict Injury Risk"):
    # Call the prediction function
    prediction, risk_percentage, injury_type, solution, prediction_proba = predict_injury_risk(
        age, match_load, training_load, fatigue_level, bmi, role, previous_injury, environmental_factor, diet_quality
    )

    # Display prediction results using HTML template
    with open("style.html") as f:
        html_template = f.read().format(prediction=prediction, risk_percentage=risk_percentage, injury_type=injury_type, solution=solution)
        st.markdown(html_template, unsafe_allow_html=True)

    # Display the predicted probabilities
    st.write(f"Predicted probabilities for each risk level (Low, Medium, High):")
    for i, prob in zip(label_encoder.classes_, prediction_proba):
        st.write(f"{i}: {prob * 100:.2f}%")
