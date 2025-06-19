# Week 1 Project - Water Quality Prediction
 
This project focuses on predicting key water quality parameters using historical data from 2000 to 2021. It employs a MultiOutput Regression model with Random Forests to estimate six major pollutants: **O2, NO3, NO2, SO4, PO4, and CL** based on temporal features such as year, month, and station ID.

---

## 📁 Dataset

- **File**: `PB_All_2000_2021.csv`
- **Description**: Water quality data collected over 21 years from various monitoring stations.
- **Key Columns**:
  - `date` – Date of sampling
  - `id` – Monitoring station identifier
  - `O2`, `NO3`, `NO2`, `SO4`, `PO4`, `CL` – Pollutant levels (target variables)

---

## 🚀 Project Objective

To build a predictive model that can estimate multiple water quality parameters simultaneously using limited temporal features (year, month, and station ID), helping stakeholders to:
- Anticipate pollutant levels
- Optimize monitoring resources
- Enable early warnings for environmental issues

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data handling
  - `scikit-learn` – Machine learning and model evaluation
  - `matplotlib`, `seaborn` – Visualization
  - `joblib` – Model persistence

---

## 🧪 Model Details

- **Model**: `MultiOutputRegressor` wrapping `RandomForestRegressor`
- **Hyperparameter Tuning**: Done using `GridSearchCV` for:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]

- **Features Used**:
  - `year`, `month`, `id` (station ID)

- **Target Variables**:
  - `O2`, `NO3`, `NO2`, `SO4`, `PO4`, `CL`

---

## 📈 Evaluation Metrics

The model was evaluated using:
- **R² Score**: Measures goodness of fit
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
  
## ✅ What The Plot Shows:
- It compares the actual vs predicted values for the pollutant O2, NO3, NO2, SO4, PO4, CL.
- The blue line represents the true values.
-  The orange line represents the predicted values from your model.
