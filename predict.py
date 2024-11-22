import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")

print(data.head())

# Dataset Information:
# This dataset includes the following features:

# age: The age of the patient.
# sex: Gender of the patient (0: female, 1: male).
# cp: Type of chest pain.
# trestbps: Resting blood pressure.
# chol: Serum cholesterol.
# fbs: Fasting blood sugar > 120 mg/dl.
# restecg: Resting electrocardiographic results.
# thalach: Maximum heart rate achieved.
# exang: Exercise induced angina.
# oldpeak: ST depression induced by exercise relative to rest

# split dataset 
X, y = data.drop(columns=["target"]), data["target"]

print("X is: ", X)
print("y is : ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# making predictions
ypred = model.predict(X_test)

print("predictions vs acutal predicitons are : ")
print(ypred, y_test)

accuracy = accuracy_score(y_test, ypred)
print("Accuracy Score of Logistic regression model is: ", accuracy)

