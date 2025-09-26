# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Load and preprocess the dataset
# Assuming you have a dataset named 'data.csv' with columns 'sorting_time' and 'delivery_time' for delivery time prediction
# Assuming you have a dataset named 'salary_data.csv' with columns 'YearsExperience' and 'Salary' for salary hike prediction

data = pd.read_csv('C:\Users\Admin')
salary_data = pd.read_csv('salary_data.csv')

# Step 3: Perform exploratory data analysis (EDA)
print(data.head())
print(salary_data.head())

# Step 4: Split the dataset into training and testing sets
X_delivery = data['sorting_time'].values.reshape(-1, 1)
y_delivery = data['delivery_time'].values

X_salary = salary_data['YearsExperience'].values.reshape(-1, 1)
y_salary = salary_data['Salary'].values

X_delivery_train, X_delivery_test, y_delivery_train, y_delivery_test = train_test_split(X_delivery, y_delivery, test_size=0.2, random_state=42)
X_salary_train, X_salary_test, y_salary_train, y_salary_test = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

# Step 5: Train the linear regression model
delivery_model = LinearRegression()
delivery_model.fit(X_delivery_train, y_delivery_train)

salary_model = LinearRegression()
salary_model.fit(X_salary_train, y_salary_train)

# Step 6: Evaluate the model using appropriate metrics
delivery_train_predictions = delivery_model.predict(X_delivery_train)
delivery_test_predictions = delivery_model.predict(X_delivery_test)

print("Mean Squared Error (MSE) for Delivery Time Prediction:")
print("Training Set:", mean_squared_error(y_delivery_train, delivery_train_predictions))
print("Testing Set:", mean_squared_error(y_delivery_test, delivery_test_predictions))

salary_train_predictions = salary_model.predict(X_salary_train)
salary_test_predictions = salary_model.predict(X_salary_test)

print("\nMean Squared Error (MSE) for Salary Hike Prediction:")
print("Training Set:", mean_squared_error(y_salary_train, salary_train_predictions))
print("Testing Set:", mean_squared_error(y_salary_test, salary_test_predictions))
