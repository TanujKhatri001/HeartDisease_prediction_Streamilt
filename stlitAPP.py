import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")

X, y = data.drop(columns=["target"]), data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# making predictions
ypred = model.predict(X_test)

accuracy = accuracy_score(y_test, ypred)

# input features
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3]) 
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120) 
chol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1]) 
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2]) 
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150) 
exang = st.selectbox("Exercise Induced Angina", options=[0, 1]) 
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2]) 
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0) 
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])
# Prediction 
input_data = pd.DataFrame({ "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps], "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach], "exang": [exang], "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal] })

if st.button("Predict"):
    prediction = model.predict(input_data)[0] 
    st.write(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}") 
    st.write(f"Model Accuracy: {accuracy:.2f}")