import pandas as pd
import pickle
import os

# Paths
model_path = os.path.join("data", "house_model.pkl")
columns_path = os.path.join("data", "columns.pkl")
test_path = os.path.join("data", "test.csv")
output_path = os.path.join("data", "predictions.csv")

# Load the trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load columns from training
with open(columns_path, "rb") as f:
    train_columns = pickle.load(f)

# Load test data
df_test = pd.read_csv(test_path)

# Fill missing values
df_test = df_test.fillna(0)

# Convert categorical columns to numeric
df_test = pd.get_dummies(df_test, drop_first=True)

# Add missing columns and reorder to match training
for col in train_columns:
    if col not in df_test.columns:
        df_test[col] = 0

df_test = df_test[train_columns]

# Predict house prices
predictions = model.predict(df_test)
df_test["Predicted_Price"] = predictions

# Save predictions
df_test.to_csv(output_path, index=False)
print(f"Predictions done! Check '{output_path}'")
