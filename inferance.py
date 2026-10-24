import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# 1. Paths
train_path = Path.cwd() / "Data" / "train.csv"
test_path  = Path.cwd() / "Data" / "test.csv"
model_dir  = Path.cwd() / "models"

print("Loading test dataset...")
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# Save 'id' column for the final submission
test_ids = test['id']

# 2. Symmetric Feature Engineering (Exact same logic as training)
test_features = test[['IsActiveMember', 'HasCrCard', 'NumOfProducts', 'Geography', 'Gender', 'EstimatedSalary', 'Balance', 'Age', 'CreditScore']].copy()

# Add the 5 domain features
test_features['BalanceToCreditScore'] = test_features['Balance'] / (test_features['CreditScore'] + 1)
test_features['ProducTimeActivity']  = test_features['IsActiveMember'] * test_features['NumOfProducts']
test_features['ZeroBalance']          = (test_features['Balance'] == 0).astype(int)
test_features['Age_Risk_Group']       = ((test_features['Age'] >= 45) & (test_features['Age'] <= 65)).astype(int)
test_features['TotalServices']        = test_features['NumOfProducts'] + test_features['HasCrCard'] + test_features['IsActiveMember']

# Surname Frequency: Learned from training data (fill unseen test names with 1)
surname_counts = train['Surname'].value_counts()
test_features['SurnameFrequency'] = test['Surname'].map(surname_counts).fillna(1)

# One-Hot Encoding
test_features = pd.get_dummies(test_features, columns=["Geography", "Gender"], drop_first=True)

# 3. Model Loading & 15-Model Ensembling
n_test = len(test)
final_preds = np.zeros(n_test)
n_folds = 5

print(f"Loading 15 saved models from {model_dir.name}/ and predicting...")

for fold in range(n_folds):
    print(f"  Predicting with Fold {fold + 1} models...")
    
    # Load the 3 models for this fold
    xgb = joblib.load(model_dir / f"xgb_fold_{fold}.joblib")
    lgb = joblib.load(model_dir / f"lgb_fold_{fold}.joblib")
    rf  = joblib.load(model_dir / f"rf_fold_{fold}.joblib")
    
    # Predict probabilities for this fold
    p_xgb = xgb.predict_proba(test_features)[:, 1]
    p_lgb = lgb.predict_proba(test_features)[:, 1]
    p_rf  = rf.predict_proba(test_features)[:, 1]
    
    # Weighted blend for this fold (50% XGB, 35% LGB, 15% RF)
    fold_pred = (p_xgb * 0.50) + (p_lgb * 0.35) + (p_rf * 0.15)
    
    # Accumulate 1/5th to the final predictions
    final_preds += fold_pred / n_folds

# 4. Create and Save the Submission File
submission = pd.DataFrame({
    'id': test_ids,
    'Exited': final_preds
})

submission_path = Path.cwd() / "submission.csv"
submission.to_csv(submission_path, index=False)

print("\n" + "="*45)
print(f"✅ Submission successfully saved to: {submission_path.name}")
print(f"Total rows: {len(submission)}")
print(f"Predicted Churn Rate (Mean): {submission['Exited'].mean():.4f}")
print("First 5 rows:")
print(submission.head())
print("="*45)