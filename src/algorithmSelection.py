import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src import preProcess

target_column = 'log_pSat_Pa'

train_data = pd.read_csv('../resources/train.csv')
test_data = pd.read_csv('../resources/test.csv')
test_data1 = test_data.copy()

X_train, y_train, X_test = preProcess.preProcessDataset(train_data, test_data, target_column, False, True, True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define regression algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),
    "Support Vector Regressor": make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
}

# Function to evaluate models
results = {}
for name, model in models.items():
    # Cross-validation for better estimation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()  # Convert to positive MSE
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"CV MSE": mean_score, "Test MSE": mse, "MAE": mae, "R2": r2}

# Display results
results_df = pd.DataFrame(results).T
results_df.sort_values(by="Test MSE", inplace=True)
print(results_df)
