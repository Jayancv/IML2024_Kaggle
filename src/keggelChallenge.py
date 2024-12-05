from datetime import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import preProcess

target_column = 'log_pSat_Pa'

train_data = pd.read_csv('../resources/train.csv')
test_data = pd.read_csv('../resources/test.csv')
test_data1 = test_data.copy()

X_train, y_train, X_test = preProcess.preProcessDataset(train_data, test_data, target_column, True, True, False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the SVR model with RBF kernel
svr_rbf = SVR(kernel='rbf', C=3, epsilon=0.45, gamma='auto')
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
