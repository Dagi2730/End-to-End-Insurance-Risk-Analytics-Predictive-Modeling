# AlphaCare Insurance Premium Prediction – Deployment Branch

## Project Overview
This branch contains the **deployment-ready system** for AlphaCare Insurance Premium Prediction. It includes a **FastAPI backend** for serving model predictions and a **Streamlit dashboard** for interactive risk analytics. The goal is to provide real-time premium and risk prediction for new insurance policies, supporting data-driven decision-making.

Key features:

- REST API for **risk prediction** using pre-trained XGBoost model
- Interactive **Streamlit dashboard** for policy input and portfolio overview
- Integration of **preprocessing pipeline** for consistent model input
- Ready for **local or cloud deployment**

---

## API Usage
**Endpoint:** `POST /predict`  

**Request Example (JSON):**
```json
{
  "Age": 35,
  "Gender": "Male",
  "VehicleType": "SUV",
  "Province": "Gauteng",
  "CustomValueEstimate": 25000.0,
  "TotalPremium": 1800.0,
  "ZipCode": 4122,
  "VehicleIntroDate": "2020-01-01"
}

Response Example:

{
  "predicted_risk": 9866.6357421875
}


Run API locally:

cd api
python main.py


Access at http://0.0.0.0:8000

Dashboard Usage

Launch dashboard:

cd dashboard
python run_dashboard.py


Default port: 8501

Enter policy details in the sidebar to get predicted risk level.