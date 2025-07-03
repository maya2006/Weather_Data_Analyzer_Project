import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("weather_data.csv")

# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Normalize Data (simple normalization for temperature)
df["temperature"] = (df["temperature"] - df["temperature"].mean()) / df["temperature"].std()

# Line Chart: Temperature trends
plt.figure(figsize=(8, 4))
plt.plot(df["year"], df["temperature"], marker="o", color="red")
plt.title("Temperature Trends Over Years")
plt.xlabel("Year")
plt.ylabel("Normalized Temperature")
plt.grid(True)
plt.tight_layout()
plt.savefig("temperature_trend.png")
plt.close()

# Bar Graph: Rainfall
plt.figure(figsize=(8, 4))
plt.bar(df["year"], df["rainfall"], color="blue")
plt.title("Yearly Rainfall Distribution")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig("rainfall_distribution.png")
plt.close()

# Scatter Plot: Humidity vs Temperature
plt.figure(figsize=(6, 4))
plt.scatter(df["humidity"], df["temperature"], color="purple")
plt.title("Humidity vs Temperature Correlation")
plt.xlabel("Humidity (%)")
plt.ylabel("Normalized Temperature")
plt.tight_layout()
plt.savefig("humidity_vs_temperature.png")
plt.close()

# Predictive Modelling: Linear Regression
X = df[["year"]]
y = df["temperature"]
model = LinearRegression()
model.fit(X, y)
future_years = np.array(range(2021, 2026)).reshape(-1, 1)
predictions = model.predict(future_years)

# Plot prediction
plt.figure(figsize=(8, 4))
plt.scatter(df["year"], df["temperature"], color="green", label="Historical Data")
plt.plot(future_years, predictions, color="orange", label="Prediction (Trend Line)")
plt.title("Temperature Prediction for Next Years")
plt.xlabel("Year")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("temperature_prediction.png")
plt.close()

# Print Statistical Summary
print("Mean Temperature:", round(df["temperature"].mean(), 2))
print("Mean Humidity:", round(df["humidity"].mean(), 2))
print("Mean Rainfall:", round(df["rainfall"].mean(), 2))

# Evaluate model
mse = mean_squared_error(y, model.predict(X))
rmse = np.sqrt(mse)
print("Model Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
