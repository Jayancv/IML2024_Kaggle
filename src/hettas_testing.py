from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import preProcess2

target_column = 'log_pSat_Pa'

train_data = pd.read_csv('../resources/train.csv')#[14000:17000]
test_data = pd.read_csv('../resources/test.csv')
test_data1 = test_data.copy()

X_train, y_train, X_test = preProcess2.preProcessDataset(train_data, test_data, target_column, False, True, False)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Define and train the SVR model with RBF kernel
svr_rbf = SVR(kernel='rbf', C=100, epsilon=0.8, gamma=0.005)

# scores = cross_val_score(svr_rbf, X_train_scaled, y_train, cv=10)
# # print("Cross-validation scores:", scores)
# print("Mean score (SVR):", np.mean(scores))


svr_rbf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svr_rbf.predict(X_test_scaled)

ols_submission = pd.DataFrame({
    'ID': test_data1['ID'],
    'TARGET': y_pred
})
print(ols_submission)
current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ols_submission.to_csv(f'../resources/sample_submission_{current_timestamp}.csv', index=False)
