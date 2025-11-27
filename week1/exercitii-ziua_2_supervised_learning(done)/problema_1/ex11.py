import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib
file_path = "train_data.csv"
df = pd.read_csv(file_path)
df_train = df
X = df.drop(columns=['Price', 'ID'])
y = df['Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

file_path = "test_data.csv"
df_test = pd.read_csv(file_path)
X_test = df_test.drop(columns=['ID'])
models = {"RandomForest": RandomForestRegressor(n_estimators=100,random_state=42),
"LinearRegression": LinearRegression(),
"KNNRegressor": KNeighborsRegressor(n_neighbors=5)
}
os.makedirs("models", exist_ok=True)
for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}")







mean_train = {}
std_train = {}
for col in X_train.columns:
    mean_train[col] = sum(X_train[col]) / len(X_train[col])
for col in X_train.columns:
    squared_diff_sum = 0
    for value in X_train[col]:
        squared_diff_sum += (value - mean_train[col]) ** 2
    std_train[col] = (squared_diff_sum / len(X_train[col])) ** 0.5

X_train_scaled = pd.DataFrame()
X_val_scaled = pd.DataFrame()
X_test_scaled = pd.DataFrame()
for col in X_train.columns:
    X_train_scaled[col] = (X_train[col] - mean_train[col]) / std_train[col]
    X_val_scaled[col] = (X_val[col] - mean_train[col]) / std_train[col]
    X_test_scaled[col] = (X_test[col] - mean_train[col]) / std_train[col]




X_train_scaled_with_y = X_train_scaled.copy()
X_train_scaled_with_y['Price'] = y_train
correlations = X_train_scaled_with_y.corr()
correlations_with_y = correlations['Price'].abs().drop('Price')
sorted_correlations_with_y = correlations_with_y.sort_values(ascending=False)
print(sorted_correlations_with_y)
best_features = sorted_correlations_with_y.index[:10]
X_train_scaled = X_train_scaled[best_features]
X_test_scaled = X_test_scaled[best_features]
X_val_scaled = X_val_scaled[best_features]

models = {

"RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
"LinearRegression": LinearRegression(),
"KNNRegressor": KNeighborsRegressor(n_neighbors=5)}

for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, yx_pred)
    print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}")