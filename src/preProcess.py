import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter

target_column = 'log_pSat_Pa'

# Load the data
train_data = pd.read_csv('../resources/train.csv')
test_data = pd.read_csv('../resources/test.csv')


def preProcessDataset(train_data, test_data, target_column):
    # Drop unnecessary columns
    train_data.drop(columns=['ID'], inplace=True)
    test_data.drop(columns=['ID'], inplace=True)

    data = train_data.copy()

    # Separate features and target variable
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]

    features = feature_selection(data, target_column)
    print(features)
    X = X[features]
    test_X = test_data[features]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    # categorical_cols = ['parentspecies']  # Update based on dataset
    categorical_cols = []  # Update based on dataset

    # Fill missing values for numerical columns with their mean
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())
    test_X[numerical_cols] = test_X[numerical_cols].fillna(test_data[numerical_cols].mean())

    # Fill missing values for categorical columns with their mode (most frequent value)
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply preprocessing
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    data_transformed = pipeline.fit_transform(data)
    X_train_transformed = pipeline.fit_transform(X)
    test_data_transformed = pipeline.fit_transform(test_X)

    transformed_column_names = preprocessor.get_feature_names_out()
    cleaned_column_names = [name.replace('num__', '').replace('cat__', '') for name in transformed_column_names]

    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=cleaned_column_names)
    test_data_transformed_df = pd.DataFrame(test_data_transformed, columns=cleaned_column_names)

    return X_train_transformed_df[features], y, test_data_transformed_df[features]


def feature_selection(train_data, target_column):
    data = train_data
    data1 = data.copy()

    # Correlation with target variable
    data_corr = data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = data_corr.corr()
    correlation_with_target = correlation_matrix[target_column].drop(target_column)
    corr_selected_features = correlation_with_target[correlation_with_target.abs() >= 0.3].index
    print("Correlation : ")
    print(corr_selected_features)

    # Remove low-variance features
    selector = VarianceThreshold(threshold=0.2)
    X1 = data1.drop(columns=[target_column])
    y1 = data1[target_column]
    X1 = X1.select_dtypes(include=['float64', 'int64'])

    selector.fit(X1)
    X_high_variance_col = X1.columns[selector.get_support()]
    print("High variance : ")
    print(X_high_variance_col)

    model = RandomForestRegressor(random_state=42)
    rfe = RFE(model, n_features_to_select=10)
    X_rfe = rfe.fit_transform(X1, y1)
    rfe_selected_features = X1.columns[rfe.support_]
    print("Random RFE : ")
    print(rfe_selected_features)

    model2 = RandomForestRegressor(random_state=42)
    model2.fit(X1, y1)
    importances = model2.feature_importances_
    rfr_important_features = X1.columns[importances > 0.01]
    print("Random forest : ")
    print(rfr_important_features)

    combined_list = list(
        set(corr_selected_features.tolist() + X_high_variance_col.tolist() + rfe_selected_features.tolist()
            + rfr_important_features.tolist()))
    print(combined_list)

    final_features = set(corr_selected_features.tolist()) | set(X_high_variance_col.tolist()) | set(
        rfe_selected_features.tolist()) | set(rfr_important_features.tolist())

    print("Combined Features:", final_features)

    feature_sets = [corr_selected_features, X_high_variance_col, rfe_selected_features, rfr_important_features]
    all_features = [feature for feature_set in feature_sets for feature in feature_set]
    feature_counts = Counter(all_features)
    # Sort features by occurrence
    sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
    print("Feature Importance by Count:", sorted_features)

    feature_importance = sorted_features

    # Filter features with importance higher than 1
    important_features = [feature for feature, importance in feature_importance if importance > 1]

    print("Features with Importance Higher Than 1:")
    print(important_features)

    final_features = important_features
    return final_features
