# Bank Customer Churn Prediction

An end-to-end machine learning pipeline for predicting customer attrition in retail banking, developed for the [Kaggle Bank Churn Prediction Competition](https://www.kaggle.com/competitions/churn-competition-pdc/overview).

The project focuses on leak-free validation, hypothesis-driven feature engineering, Bayesian hyperparameter optimization, and multi-model out-of-fold ensembling.

---

## 📌 Problem Overview

* **Task:** Binary classification predicting whether a bank customer will exit (`Exited = 1`) or stay (`Exited = 0`).
* **Evaluation Metric:** **ROC-AUC** (Receiver Operating Characteristic - Area Under the Curve) to evaluate ranking quality independently of decision thresholds.
* **Dataset Characteristics:**
  * **Train Set:** 15,000 samples × 14 features
  * **Test Set:** 10,000 samples × 13 features
  * **Target Distribution:** ~79.4% retained, ~20.6% churned (imbalanced classification).

---

## 🏗 Pipeline Architecture

```text
Data Ingestion (train.csv / test.csv)
       │
       ▼
Exploratory Data Analysis (EDA)
  • Class balance verification
  • Non-linear behavioral analysis (NumOfProducts thresholding)
       │
       ▼
Feature Engineering & Selection
  • Interaction terms (ProducTimeActivity = IsActiveMember * NumOfProducts)
  • Financial ratios (BalanceToCreditScore)
  • Demographic clustering (Surname frequency encoding, Age risk grouping)
  • Permutation Feature Importance pruning
       │
       ▼
Local Validation Setup
  • Stratified 5-Fold Cross-Validation (preserves 80/20 target distribution)
       │
       ▼
Hyperparameter Optimization
  • Bayesian Optimization via Optuna (Tree-structured Parzen Estimator)
       │
       ▼
Out-Of-Fold Ensembling (15 Models)
  • XGBoost (50%) + HistGradientBoosting (35%) + Random Forest (15%)
       │
       ▼
Inference & Submission (submission.csv)
```

---

## 🔬 Key Engineering & Modeling Highlights

### 1. Exploratory Insights & Non-Linear Signals
* **Product Thresholding:** Analysis revealed strong non-linear behavior in `NumOfProducts`: customers with 2 products exhibited the lowest churn (~4.5%), whereas customers with 3 or 4 products churned at ~97%–100%.
* **Model Families:** Linear baselines (Logistic Regression) struggled on raw inputs (~0.78 ROC-AUC) due to an inability to separate these non-linear thresholds without polynomial expansion, establishing Gradient Boosted Trees as the primary architecture.

### 2. Feature Engineering & Permutation Diagnostics
* **`ProducTimeActivity`:** ($NumOfProducts \times IsActiveMember$) — Gated product engagement feature; emerged as one of the top predictive drivers in permutation importance.
* **`BalanceToCreditScore`:** ($Balance / [CreditScore + 1]$) — Captured financial leverage better than raw CreditScore.
* **`SurnameFrequency`:** Extracted demographic family size signals from text strings without target leakage.
* **Permutation Importance:** Measured validation score drops upon column shuffling to identify and discard noisy features (e.g., `BalanceToSalary` was removed after yielding negative importance).

### 3. Bayesian Optimization (Optuna TPE)
Hyperparameters for `XGBClassifier` were tuned using Tree-structured Parzen Estimators (TPE) over 30 trials on GPU (`CUDA`), converging on:
* `max_depth: 4` (constrained to avoid the overfitting seen at default depth 6)
* `learning_rate: ~0.039` (log-uniform search for steady convergence)
* `subsample: 0.77` & `colsample_bytree: 0.71` (stochastic row and feature bagging to reduce variance)
* `n_estimators: 191`

### 4. 15-Model Out-Of-Fold Ensemble
To eliminate structural bias from any single algorithm, predictions were blended across three diverse model families across 5 folds:
* **XGBoost (50% weight):** Fast exact-split gradient boosting.
* **HistGradientBoosting (35% weight):** Scikit-learn's LightGBM-style histogram-based tree growth.
* **Random Forest (15% weight):** Bootstrap aggregation for boundary smoothing.

---

## 📊 Iterative Progression

| Stage | Model Configuration | Local 5-Fold CV ROC-AUC |
| :--- | :--- | :---: |
| **Baseline** | Logistic Regression (Raw Features) | ~0.7820 |
| **Baseline** | Default GradientBoostingClassifier | ~0.9265 |
| **Feature Engineering** | Engineered Ratios + Interaction Terms | ~0.9304 |
| **Tuned Single Model** | GPU XGBoost (Bayesian TPE Tuned) | ~0.9328 |
| **Final Ensemble** | 15-Model Weighted Blend (XGB + HistGB + RF) | **0.9331** |

---

## 📁 Repository Structure

```text
├── Data/
│   ├── train.csv               # 15,000 labeled samples
│   ├── test.csv                # 10,000 unlabeled evaluation samples
│   └── sample_submission.csv   # Target submission format
├── main.py                     # Initial EDA, baseline models, and Permutation Importance
├── train_ensemble.py           # 5-fold cross-validation, feature pipeline, and model serialization
├── inference.py                # Test set preprocessing and 15-model ensemble inference
├── submission.csv              # Final predicted probabilities
├── .gitignore
└── README.md
```

---

## 🚀 How to Reproduce

### 1. Requirements
Ensure Python 3.10+ is installed along with the dependencies:
```bash
pip install pandas numpy scikit-learn xgboost optuna joblib
```

### 2. Train and Serialize Models
Run the multi-model 5-fold training pipeline:
```bash
python train_ensemble.py
```
This generates the Out-Of-Fold metrics and serializes 15 trained models into a `models/` directory.

### 3. Generate Submission
Run the inference script to evaluate `test.csv`:
```bash
python inference.py
```
This outputs `submission.csv` with verified row alignment and calibrated probability distributions.
