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