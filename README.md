# 🌦 Weather Forecasting using Time Series Analysis (Linear Regression)

## 📌 Problem Statement

This project focuses on forecasting daily average temperature in India using **Simple Linear Regression** applied to **time series weather data**. While linear regression is a basic approach, it offers an introduction to time-dependent prediction and highlights the limitations of simple models in forecasting.

---

## 📂 Dataset

- **Source**: [Kaggle - Daily Temperature of Major Cities](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities)
- **File Used**: `city_temperature.csv`
- **Features**:
  - Country, Region, City
  - Year, Month, Day
  - AvgTemperature

Only temperature records from **India** are used for analysis.

---

## ⚙️ Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib

---

## 🔍 Steps Followed

### 1. Load the Dataset
```python
data = pd.read_csv('dataset/city_temperature.csv', low_memory=False)
2. Filter and Prepare Data
Only records from India with valid average temperatures (AvgTemperature > -99) are selected.

Combine Year, Month, Day into a single datetime column.

Convert dates to ordinal format to use as a numeric feature.

3. Split Data
80% Training / 20% Validation using train_test_split.

4. Train Model
python
Copy
Edit
model = LinearRegression()
model.fit(X_train, y_train)
5. Evaluate
Mean Squared Error (MSE)

Scatter plot of actual vs predicted temperatures.

# 📈 Output Visualization
![image](https://github.com/user-attachments/assets/e3b459c7-057b-4919-813b-e0b8df4ed00d)
