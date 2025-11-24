import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"C:\Users\SAI\OneDrive\Desktop\ProjectsMl\Synthetic_5000_Energy_Consumption_With_Seasons.csv")

label_encoders = {}
categorical_cols = ["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday", "Season"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["EnergyConsumption"])
y = df["EnergyConsumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data split into 70% training and 30% testing. Files saved.")
