# hackathon_problem
Data Loading and Preprocessing:

Load training and test datasets using pd.read_csv.
Separate features (X_train, X_test) and target variables (y_train).
Define preprocessing steps including handling missing values (SimpleImputer) and encoding categorical variables (OneHotEncoder).

Model Training:

Use MultiOutputClassifier to handle the multi-label nature of the problem and train a RandomForestClassifier.
Fit the model on preprocessed training data (X_train_processed).

Model Evaluation:

Predict probabilities (predict_proba) for both xyz_vaccine and seasonal_vaccine on the test set.
Compute ROC AUC scores for both predictions using roc_auc_score.

Prepare Submission File:

Create a DataFrame (submission_df) with respondent_id, xyz_vaccine, and seasonal_vaccine probabilities.
Save the DataFrame to a CSV file (submission.csv) for submission.
