import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Example Dataset: Replace this with your actual dataset
from sklearn.datasets import make_regression
from src import preProcess

target_column = 'log_pSat_Pa'

train_data = pd.read_csv('../resources/train.csv')
test_data = pd.read_csv('../resources/test.csv')
test_data1 = test_data.copy()

X_train, y_train, X_test = preProcess.preProcessDataset(train_data, test_data, target_column, True, True, False)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale the data (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the SVR model
svr = SVR()

# Define the hyperparameters grid to search (including different kernels)
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'epsilon': [0.01, 0.1, 0.2, 0.5],  # Epsilon parameter (controls margin of error)
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Types of kernel functions
    'degree': [2, 3, 4],  # Degree of the polynomial kernel (only for 'poly')
    'gamma': ['scale', 'auto', 0.1, 1]  # Gamma (relevant for 'rbf', 'poly', and 'sigmoid')
}

# Use GridSearchCV to perform k-fold cross-validation (k=5 here)
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model on the training data
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters found by GridSearchCV
print("Best Hyperparameters: ", grid_search.best_params_)

# Use the best estimator to predict on the test data
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error: {mse}")
print(f"Test R² Score: {r2}")
