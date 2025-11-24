import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Load trained model and scaler
model = joblib.load("energy_consumption_model.pkl")
scaler = joblib.load("scaler.pkl")

# Standardize test data
X_test_scaled = scaler.transform(X_test)

# Predict energy consumption
y_pred = model.predict(X_test_scaled)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE on Test Data: {mae}")
print(f"R² Score on Test Data: {r2}")

# 📊 Regression Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption (Regression Plot)")
plt.show()
