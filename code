# phone-pey
# 📦 Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 📥 Load Dataset
 df = pd.read_csv("your_data.csv")  # Replace this with actual file
# Assuming df is already present in your workspace

# 🧹 Handle Missing Values
df['amount'].fillna(df['amount'].mean(), inplace=True)
df['count'].fillna(df['count'].median(), inplace=True)
df['transaction_type'].fillna(df['transaction_type'].mode()[0], inplace=True)

# 🧽 Handle Outliers using IQR method
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['amount'] >= Q1 - 1.5 * IQR) & (df['amount'] <= Q3 + 1.5 * IQR)]

# 🔣 Encode Categorical Variables
le = LabelEncoder()
df['transaction_type_encoded'] = le.fit_transform(df['transaction_type'])

# 🧮 Feature Engineering
df['avg_amount'] = df['amount'] / (df['count'] + 1)
df['date'] = pd.to_datetime(df['year'].astype(str) + 'Q' + df['quarter'].astype(str))

# 🔍 Feature Selection
features = ['count', 'quarter', 'transaction_type_encoded']
target = 'amount'
X = df[features]
y = df[target]

# 🔄 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚖️ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📈 Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
print("📉 Linear Regression => R²:", r2_score(y_test, y_pred_lr), " | RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

# 🌲 Random Forest Regressor with Grid Search
rf = RandomForestRegressor(random_state=42)
param_rf = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
grid_rf = GridSearchCV(rf, param_rf, cv=3, scoring='r2')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("🌳 Random Forest => R²:", r2_score(y_test, y_pred_rf), " | RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))

# 🚀 XGBoost Regressor with Grid Search
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_xgb = {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
grid_xgb = GridSearchCV(xgb_model, param_xgb, cv=3, scoring='r2')
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print("🔥 XGBoost => R²:", r2_score(y_test, y_pred_xgb), " | RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))

# 💾 Save the Best Model
joblib.dump(best_xgb, 'xgboost_phonepe_model.pkl')
print("✅ XGBoost model saved to 'xgboost_phonepe_model.pkl'")

# 🔁 Load and Predict on New Data
loaded_model = joblib.load('xgboost_phonepe_model.pkl')
sample_input = pd.DataFrame({'count': [150], 'quarter': [2], 'transaction_type_encoded': [1]})
sample_prediction = loaded_model.predict(sample_input)
print("🔮 Sample Prediction (Unseen Data): ₹", round(sample_prediction[0], 2))
