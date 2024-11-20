from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import preProcess

target_column = 'log_pSat_Pa'

train_data = pd.read_csv('../resources/train.csv')
test_data = pd.read_csv('../resources/test.csv')
test_data1 = test_data.copy()

X_train, y_train, X_test = preProcess.preProcessDataset(train_data, test_data, target_column)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
ols_submission = pd.DataFrame({
    'ID': test_data1['ID'],
    'TARGET': y_pred
})
print(ols_submission)
current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ols_submission.to_csv(f'../resources/sample_submission_{current_timestamp}.csv', index=False)
