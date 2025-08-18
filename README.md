# Task 4: Predictive Modeling for Risk-Based Premiums

## Project Overview
Task 4 focuses on developing and evaluating **predictive models** that form the core of a **dynamic, risk-based insurance pricing system**. The goal is to predict **claim severity** and optimize **premiums**, enabling data-driven, financially sound decisions for insurance pricing.

This task includes:
- Claim Severity Prediction (Risk Model)
- Premium Optimization (Pricing Framework)
- Statistical Modeling & Interpretability

---

## Data Summary

**Claims Subset Shape:** `(8415, 9)`

| PolicyID   | Province        | ZipCode | Gender | VehicleType | VehicleIntroDate               | CustomValueEstimate | TotalPremium  | TransactionMonth               |
|-----------|----------------|---------|--------|-------------|--------------------------------|------------------|---------------|-------------------------------|
| POL_000000 | KwaZulu-Natal   | 4122    | Female | SUV         | 2000-01-01 00:00:00           | 248845.50         | 16076.84      | 2014-02-01 00:00:00           |
| POL_000001 | Eastern Cape    | 9167    | Male   | Hatchback   | 2000-01-01 18:24:35           | 210532.78         | 19823.19      | 2014-02-01 01:22:57           |
| POL_000002 | Gauteng         | 6190    | Female | Sedan       | 2000-01-02 12:49:10           | 169778.61         | 15129.38      | 2014-02-01 02:45:54           |
| POL_000004 | KwaZulu-Natal   | 8727    | Male   | Hatchback   | 2000-01-04 01:38:21           | 162682.61         | 13500.24      | 2014-02-01 05:31:48           |
| POL_000005 | Eastern Cape    | 2797    | Male   | Truck       | 2000-01-04 20:02:57           | 241486.69         | 11442.04      | 2014-02-01 06:54:45           |

**Train-Test Split:**  
- Training set: `(6732, 9)`  
- Test set: `(1683, 9)`  

---

## Modeling Goals

### Claim Severity Prediction (Risk Model)
- **Objective:** Predict `TotalClaims` for policies with claims.
- **Target Variable:** `TotalClaims` (subset of claims > 0)
- **Evaluation Metrics:** RMSE (Root Mean Squared Error), R²
- **Purpose:** Estimate financial liability per policy

### Premium Optimization (Pricing Framework)
- **Objective:** Predict appropriate insurance premium
- **Advanced Approach:**  
  Premium = (Predicted Probability of Claim × Predicted Claim Severity) + Expense Loading + Profit Margin
- **Evaluation Metrics:** RMSE, R² for regression; Accuracy, Precision, Recall, F1-Score for claim probability

---

## Model Training & Evaluation

### Claim Severity Models
- ✅ LinearRegression trained successfully  
- ✅ RandomForest trained successfully  
- ✅ XGBoost trained successfully  

| Model               | RMSE     | R²       |
|--------------------|----------|----------|
| LinearRegression    | 6716.93 | 0.0213   |
| RandomForest        | 6988.94 | -0.0596  |
| XGBoost             | 6763.29 | 0.0077   |

**Baseline Metrics:**  
- RMSE: 6790.18  
- R²: -0.0002  

**Best XGBoost Parameters:**  
```json
{'subsample': 1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
