from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import preProcess

target_column = 'log_pSat_Pa'

train_data = pd.read_csv('../resources/train.csv')
test_data = pd.read_csv('../resources/test.csv')
test_data1 = test_data.copy()

X_train, y_train, X_test = preProcess.preProcessDataset(train_data, test_data, target_column, True, True, False)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# model = RandomForestRegressor()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# print(y_pred)
# ols_submission = pd.DataFrame({
#     'ID': test_data1['ID'],
#     'TARGET': y_pred
# })
# print(ols_submission)
# current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# ols_submission.to_csv(f'../resources/sample_submission_{current_timestamp}.csv', index=False)
for i in [0.425, 0.45, 0.475, 0.5]:
    for j in [1, 1.5, 2, 3, 5]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define and train the SVR model with RBF kernel
        svr_rbf = SVR(kernel='rbf', C=j, epsilon=i, gamma='auto')
        svr_rbf.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred = svr_rbf.predict(X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Test Mean Squared Error: epsilon:{i}, C:{j} :{mse}")
