import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# 1. Setup Directories
data_path = Path.cwd() / "Data" / "train.csv"
model_dir = Path.cwd() / "models"
model_dir.mkdir(exist_ok=True)  # Creates 'models/' folder automatically

train = pd.read_csv(data_path)
y = train['Exited']

# 2. Complete Feature Engineering Pipeline
x = train[['IsActiveMember', 'HasCrCard', 'NumOfProducts', 'Geography', 'Gender', 'EstimatedSalary', 'Balance', 'Age', 'CreditScore']].copy()

# Domain Features & Ratios
x['BalanceToCreditScore'] = x['Balance'] / (x['CreditScore'] + 1)
x['ProducTimeActivity'] = x['IsActiveMember'] * x['NumOfProducts']
x['ZeroBalance'] = (x['Balance'] == 0).astype(int)
x['Age_Risk_Group'] = ((x['Age'] >= 45) & (x['Age'] <= 65)).astype(int)
x['TotalServices'] = x['NumOfProducts'] + x['HasCrCard'] + x['IsActiveMember']

# Surname Frequency
surname_counts = train['Surname'].value_counts()
x['SurnameFrequency'] = train['Surname'].map(surname_counts)

# One-Hot Encoding
x = pd.get_dummies(x, columns=["Geography", "Gender"], drop_first=True)

# 3. Setup Out-Of-Fold Prediction Arrays
n_samples = len(train)
oof_xgb = np.zeros(n_samples)
oof_lgb = np.zeros(n_samples)
oof_rf  = np.zeros(n_samples)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting 5-Fold Training for 3 Model Architectures...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"--- FOLD {fold + 1} ---")
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ================= MODEL 1: Tuned XGBoost (GPU) =================
    model_xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.039,
        subsample=0.77,
        colsample_bytree=0.71,
        min_child_weight=2,
        n_estimators=190,
        random_state=42,
        device='cuda'
    )
    model_xgb.fit(x_train, y_train)
    oof_xgb[val_idx] = model_xgb.predict_proba(x_val)[:, 1]
    joblib.dump(model_xgb, model_dir / f"xgb_fold_{fold}.joblib")

    # ================= MODEL 2: HistGradientBoosting (LightGBM style) =================
    model_lgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
    model_lgb.fit(x_train, y_train)
    oof_lgb[val_idx] = model_lgb.predict_proba(x_val)[:, 1]
    joblib.dump(model_lgb, model_dir / f"lgb_fold_{fold}.joblib")

    # ================= MODEL 3: Random Forest =================
    model_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(x_train, y_train)
    oof_rf[val_idx] = model_rf.predict_proba(x_val)[:, 1]
    joblib.dump(model_rf, model_dir / f"rf_fold_{fold}.joblib")

    # Fold Blend Score
    fold_blend = (oof_xgb[val_idx] * 0.50) + (oof_lgb[val_idx] * 0.35) + (oof_rf[val_idx] * 0.15)
    print(f"  XGB AUC:   {roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")
    print(f"  LGB AUC:   {roc_auc_score(y_val, oof_lgb[val_idx]):.4f}")
    print(f"  RF AUC:    {roc_auc_score(y_val, oof_rf[val_idx]):.4f}")
    print(f"  Blend AUC: {roc_auc_score(y_val, fold_blend):.4f}\n")

# 4. Final Out-Of-Fold Summary
auc_xgb = roc_auc_score(y, oof_xgb)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_rf  = roc_auc_score(y, oof_rf)

# Weighted Ensemble Blend: 50% XGBoost + 35% HistGB + 15% Random Forest
oof_blend = (oof_xgb * 0.50) + (oof_lgb * 0.35) + (oof_rf * 0.15)
auc_blend = roc_auc_score(y, oof_blend)

print("="*45)
print("FINAL OUT-OF-FOLD (OOF) SCORES:")
print(f"  XGBoost OOF ROC-AUC:             {auc_xgb:.5f}")
print(f"  HistGradientBoosting OOF ROC-AUC:{auc_lgb:.5f}")
print(f"  Random Forest OOF ROC-AUC:       {auc_rf:.5f}")
print("-" * 45)
print(f"  🏆 BLENDED ENSEMBLE OOF ROC-AUC:  {auc_blend:.5f}")
print("="*45)
print(f"All 15 models saved to: {model_dir.resolve()}")