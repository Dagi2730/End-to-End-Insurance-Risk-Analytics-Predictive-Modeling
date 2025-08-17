# AlphaCare Insurance Premium Prediction

## Project Overview
This project implements an end-to-end predictive analytics pipeline for **insurance premium prediction** using historical policy and claims data. The primary goal is to help AlphaCare Insurance Solutions optimize pricing, assess risk, and improve decision-making using machine learning.

The pipeline includes:

- Advanced **feature engineering** (risk-based, vehicle, geographic, demographic, temporal, and interaction features)
- Comprehensive **data preprocessing and feature expansion**
- **Model training and evaluation** using multiple regression algorithms
- **Model interpretability** via SHAP for feature importance analysis
- Final **deployment-ready model** with prediction template

---

## Dataset
- **Samples:** 10,000 policies
- **Original Features:** 10
- **Engineered Features:** 24 (14 new features added)
- **Primary Target Variable:** `TotalPremium`
- **Target Statistics:**
  - Mean: 16,076.84
  - Range: 16,076.84 – 16,076.84

---

## Feature Engineering
- Risk-based features (`RiskScore`, `ClaimRate`, etc.)
- Vehicle-related features (`HighRiskVehicle`, `VehicleType`)
- Geographic features (`ProvinceRiskScore`, `ZipCode`)
- Demographic features (`GenderRisk`)
- Temporal features (`Month`, `Quarter`)
- Interaction and statistical features

**Total features after processing:** 17 original + engineered → 10383 processed features  

---

## Model Training and Evaluation
**Models trained:**

- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- XGBoost Regressor  

**Best Model Selected:** XGBoost  
- Validation R²: 0.0009  
- Validation RMSE: 7,302.49  
- Test R²: -0.0112  
- Test RMSE: 7,441.65  
- Residual standard deviation: 7,441.21  

## Model Interpretability
**SHAP Analysis:**  
Top 15 features contributing to premium prediction:

1. `VehicleType_SUV`  
2. `TotalPremium`  
3. `CustomValueEstimate`  
4. `ZipCode_8777`  
5. `ZipCode_2080`  
6. `ZipCode_3829`  
7. `ZipCode_1948`  
8. `Province_KwaZulu-Natal`  
9. `ZipCode_5250`  
10. `ZipCode_5759`  
11. `ZipCode_8082`  
12. `ZipCode_7168`  
13. `ZipCode_3348`  
14. `ZipCode_7452`  
15. `ZipCode_7792`  

- Base prediction (expected value): 9,126.56  
- Feature contribution range: [-3,431.18, 8,469.35]  

---

## Saved Artifacts
All models and results are saved for deployment:

- **Best Model:** `../models/best_model_xgboost.joblib`  
- **Preprocessor:** `../models/preprocessor.joblib`  
- **Prediction Template:** `../models/prediction_template.py`  
- **Modeling Results:** `../results/modeling_results.json`  
- **Deployment Info:** `../models/deployment_info.json`  

**Prediction template** allows easy integration for new data:

```python
from prediction_template import load_model, predict_premium

model, preprocessor = load_model()
predictions = predict_premium(new_data, model, preprocessor)
  
  