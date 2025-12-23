# ==============================
# Diabetes Prediction using ML
# ==============================

# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# ==============================
# Data Collection & Analysis
# ==============================

# Load the dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Display basic information
print("Dataset Shape:", diabetes_dataset.shape)
print("\nFirst 5 rows:\n", diabetes_dataset.head())
print("\nStatistical Summary:\n", diabetes_dataset.describe())
print("\nOutcome Distribution:\n", diabetes_dataset['Outcome'].value_counts())


# ==============================
# Data Preprocessing
# ==============================

# Separate features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# ==============================
# Train-Test Split
# ==============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ==============================
# Model Training
# ==============================

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


# ==============================
# Model Evaluation
# ==============================

# Training accuracy
X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)
print("\nTraining Accuracy:", training_accuracy)

# Testing accuracy
X_test_prediction = classifier.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", testing_accuracy)


# ==============================
# Predictive System
# ==============================

# Feature names (must match training order)
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Example input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Standardize input
std_data = scaler.transform(input_df)

# Make prediction
prediction = classifier.predict(std_data)

# Output result
if prediction[0] == 0:
    print("\nPrediction: The person is NOT diabetic")
else:
    print("\nPrediction: The person IS diabetic")
