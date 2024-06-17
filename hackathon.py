# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate features and target variables
X_train = train_df.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y_train = train_df[['xyz_vaccine', 'seasonal_vaccine']]
X_test = test_df.drop(['respondent_id'], axis=1)

# Define categorical features for encoding
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train a multi-output Random Forest classifier
forest_clf = RandomForestClassifier(random_state=42)

multi_target_forest = MultiOutputClassifier(forest_clf, n_jobs=-1)
multi_target_forest.fit(X_train_processed, y_train)

# Predict probabilities for the test set
y_pred_proba = multi_target_forest.predict_proba(X_test_processed)

# Extract probabilities for xyz_vaccine and seasonal_vaccine
xyz_vaccine_proba = y_pred_proba[0][:, 1]
seasonal_vaccine_proba = y_pred_proba[1][:, 1]

# Calculate ROC AUC scores
xyz_vaccine_auc = roc_auc_score(y_test['xyz_vaccine'], xyz_vaccine_proba)
seasonal_vaccine_auc = roc_auc_score(y_test['seasonal_vaccine'], seasonal_vaccine_proba)

print(f"ROC AUC for xyz_vaccine: {xyz_vaccine_auc}")
print(f"ROC AUC for seasonal_vaccine: {seasonal_vaccine_auc}")

# Create submission DataFrame
submission_df = pd.DataFrame({
    'respondent_id': test_df['respondent_id'],
    'xyz_vaccine': xyz_vaccine_proba,
    'seasonal_vaccine': seasonal_vaccine_proba
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
