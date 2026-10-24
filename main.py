import numpy as np 
import pandas as pd
from pathlib import Path 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import optuna 



path = Path.cwd() / "Data" / "train.csv"

Categorical = ["Exited", "IsActiveMember", "HasCrCard", "NumOfProducts", "Gender", "Geography"]
Continous = ["EstimatedSalary", "Balance", "Tenure", "Age", "CreditScore"]

train = pd.read_csv(path)
# for item in Categorical:
#     print(train[item].value_counts(normalize=True))
 

# for item in Continous:
#     print(train[item].describe)
 
# for item in Categorical:
#     if item != "Exited":
#         print(f"\n--- Churn Rate by {item} ---")
#         print(train.groupby(item)["Exited"].agg(["count", "mean"]))


y = train['Exited']
# print(y.head)

x = train[['IsActiveMember', 'HasCrCard', 'NumOfProducts', 'Geography', 'Gender', 'EstimatedSalary', 'Balance', 'Age', 'CreditScore']]
surname_counts = train['Surname'].value_counts()
x['SurnameFrequency'] = train['Surname'].map(surname_counts)


# x['TenureToAge'] = train['Tenure'] / (x['Age'])
x['BalanceToCreditScore'] = x['Balance'] / (x['CreditScore']+1)
x['ProducTimeActivity'] = x['IsActiveMember'] * x['NumOfProducts']
x['TotalServices'] = x['NumOfProducts'] + x['HasCrCard'] + x['IsActiveMember']
x['Age_Risk_Group'] = ((x['Age'] >= 45) & (x['Age'] <= 65)).astype(int)
skf = StratifiedKFold(random_state=42, n_splits=5,shuffle=True)
x = pd.get_dummies(x,columns=["Geography", "Gender"],drop_first=True)
avg = 0

# x['BalanceToSalary'] = x['Balance'] / (x['EstimatedSalary']+1)
model = XGBClassifier( random_state=42,max_depth = 3, n_estimator=100,learning_rate = 0.08,subsample = 0.75, device = 'cuda',colsample_bytree = 0.8)



def objective(trial):
    # Ask the Bayesian sampler to suggest hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.65, 0.90),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.90),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'n_estimators': trial.suggest_int('n_estimators', 80, 200),
        'random_state': 42,
        'device': 'cuda'
    }

    scores = []
    # Use your existing Stratified 5-Fold split
    for train_idx, val_idx in skf.split(x, y):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(x_train, y_train)

        val_preds = model.predict_proba(x_val)[:, 1]
        scores.append(roc_auc_score(y_val, val_preds))

    # Optuna aims to maximize this return value
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print("\n" + "="*40)
print(f"Best Trial ROC-AUC: {study.best_value:.5f}")
print("Best Hyperparameters:")




for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*40)


















# for fold , (train_idx,val_idx) in enumerate (skf.split(x,y)):  
#     x_train,x_val = x.iloc[train_idx],x.iloc[val_idx]
#     y_train,y_val = y.iloc[train_idx],y.iloc[val_idx]

#     model.fit(x_train, y_train)

#     val_predicts = model.predict_proba(x_val)[:,1]

#     score = roc_auc_score(y_val, val_predicts)
#     avg = avg + score
#     print(f"Fold {fold + 1} ROC-AUC: {score:.4f}")


# print(f"the average is:  {avg/5:.4f} ")










# result = permutation_importance(
#         model,
#         x_val,
#         y_val,
#         scoring = 'roc_auc',
#         n_repeats = 5,
#         random_state = 42
#     )



# importance_df = pd.Series(result.importances_mean, index=x.columns).sort_values(ascending=False)
# print("\n--- Permutation Feature Importance ---")
# print(importance_df)

# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.05, 0.1],
#     'subsample': [0.7, 0.8]
# }



# grid_search = GridSearchCV(
#     estimator=XGBClassifier(random_state=42, device='cuda'),
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=skf,
#     verbose=1
# )
# grid_search.fit(x, y)

# print("Best ROC-AUC:", grid_search.best_score_)
# print("Best Parameters:", grid_search.best_params_)