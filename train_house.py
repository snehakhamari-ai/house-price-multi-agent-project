import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Paths
train_path = os.path.join("data", "train.csv")
model_path = os.path.join("data", "house_model.pkl")
columns_path = os.path.join("data", "columns.pkl")

# Load training data
df_train = pd.read_csv(train_path)

# Separate target
y_train = df_train["SalePrice"]
X_train = df_train.drop(columns=["SalePrice"])

# Fill missing values
X_train = X_train.fillna(0)

# Convert categorical columns to numeric
X_train = pd.get_dummies(X_train, drop_first=True)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Save the columns used
with open(columns_path, "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print(f"Model trained and saved as {model_path}")
