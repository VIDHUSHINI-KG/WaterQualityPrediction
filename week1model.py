# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Step 2: Load the dataset with improved date parsing
df = pd.read_csv('PB_All_2000_2021.csv', sep=';', parse_dates=['date'], dayfirst=True)

# Step 3: Basic info and missing value handling
print(df.info())
print("\nMissing values before handling:\n", df.isnull().sum())
df.dropna(inplace=True)  # You can replace with imputation if preferred

# Step 4: Feature engineering
df = df.sort_values(by=['id', 'date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Step 5: Define features and target variables
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
X = df[['year', 'month', 'id']]  # Added 'id' as a feature
y = df[pollutants]

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Model training with GridSearchCV
base_model = RandomForestRegressor(random_state=42)
model = MultiOutputRegressor(base_model)

param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Step 8: Prediction and Evaluation
y_pred = best_model.predict(X_test)

print("\nEvaluation Metrics:")
print("R2 Score:", r2_score(y_test, y_pred, multioutput='uniform_average'))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

