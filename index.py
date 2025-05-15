import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Step 1: Load data
data = pd.read_csv('city_temperature.csv', low_memory=False)

# Step 2: Filter for India and remove invalid temperatures
india_data = data[(data['Country'] == 'India') & (data['AvgTemperature'] > -99)].copy()

# Step 3: Create Date column
india_data['Date'] = pd.to_datetime(india_data[['Year', 'Month', 'Day']])

# Step 4: Select necessary columns
rel_data = india_data[['Date', 'AvgTemperature']].dropna()

# Step 5: Feature Engineering
rel_data['Date_ordinal'] = rel_data['Date'].map(pd.Timestamp.toordinal)

# Add a 7-day moving average feature to capture local trends
rel_data['MA_7'] = rel_data['AvgTemperature'].rolling(window=7).mean()
rel_data = rel_data.dropna()  # Drop rows with NaN from rolling average

# Define features and target
X = rel_data[['Date_ordinal', 'MA_7']]
y = rel_data['AvgTemperature']

# Step 6: Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Polynomial Features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Step 8: Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)

# Step 9: Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 10: Evaluate Model
predictions = model.predict(X_val_scaled)
mse = mean_squared_error(y_val, predictions)
r2 = r2_score(y_val, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 11: Plot Predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_val['Date_ordinal'], y_val, label='Actual', alpha=0.6)
plt.scatter(X_val['Date_ordinal'], predictions, label='Predicted', color='red', alpha=0.6)
plt.title('Improved Weather Forecast: Actual vs Predicted Temperatures (Polynomial Regression)')
plt.xlabel('Date (Ordinal)')
plt.ylabel('Avg Temperature')
plt.legend()
plt.tight_layout()
plt.show()

# Step 12: Predict Temperature for a Single Future Date
future_date_str = input("\nEnter a future date (YYYY-MM-DD): ")

try:
    future_date = pd.to_datetime(future_date_str)
    future_ordinal = future_date.toordinal()
    last_ma7 = rel_data['MA_7'].iloc[-1]

    future_features = np.array([[future_ordinal, last_ma7]])
    future_poly = poly.transform(future_features)
    future_scaled = scaler.transform(future_poly)

    future_temp = model.predict(future_scaled)[0]
    print(f"\nPredicted Avg Temperature for {future_date_str}: {future_temp:.2f} °F")
except Exception as e:
    print("Invalid date format or prediction failed:", e)

# Step 13: Predict Temperature for a Range of Future Dates (e.g., next 7 days)
print("\nPredicting temperature for the next 7 days:")
start_date = rel_data['Date'].max() + timedelta(days=1)
last_ma7 = rel_data['MA_7'].iloc[-1]

for i in range(7):
    day = start_date + timedelta(days=i)
    day_ordinal = day.toordinal()

    features = np.array([[day_ordinal, last_ma7]])
    features_poly = poly.transform(features)
    features_scaled = scaler.transform(features_poly)

    temp_prediction = model.predict(features_scaled)[0]
    print(f"{day.date()}: {temp_prediction:.2f} °F")
