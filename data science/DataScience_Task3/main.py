# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("car data.csv")

# %% [markdown]
# Understand the Dataset

# %%
print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.describe())

# %%
# Check Missing Values
print(df.isnull().sum())

# %% [markdown]
# Data Visualization

# %%
# Correlation Heatmap
plt.figure(figsize=(10,7))

sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.show()

# %%
# Remove Unnecessary Column
df = df.drop('Car_Name', axis=1)

# %% [markdown]
# Convert Text Data to Numbers

# %%
# Use One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# %%
# Train-Test Split
X = df.drop('Selling_Price', axis=1)

y = df['Selling_Price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# %% [markdown]
# Linear Regression

# %%
# Train Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

# %%
# Make Predictions
predictions = model.predict(X_test)

print(predictions)

# %%
# model evaluation 
from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)

print("R2 Score:", score)

# %% [markdown]
# R² Score
# 
# | Score | Meaning            |
# | ----- | ------------------ |
# | 1.0   | perfect prediction |
# | 0.9   | excellent          |
# | 0.7   | good               |
# | 0.5   | average            |
# 

# %%
import joblib

joblib.dump(model, "car_price_model.pkl")

# %% [markdown]
# Custom Prediction Example

# %%
print(X.columns)


# %% [markdown]
# Custom Prediction Example

# %%
sample = pd.DataFrame([{
    'Year': 2018,
    'Present_Price': 5.5,
    'Driven_kms': 35000,
    'Owner': 0,
    'Fuel_Type_Diesel': 0,
    'Fuel_Type_Petrol': 1,
    'Selling_type_Individual': 1,
    'Transmission_Manual': 1
}])

result = model.predict(sample)

print(result)
print("Predicted Car Price:", result[0], "Lakhs")

# %% [markdown]
# visualization comparing actual vs predicted prices.

# %%
plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.show()


