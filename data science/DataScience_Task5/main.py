# %% [markdown]
# Step 1: Data Inspection & Cleaning

# %%
import pandas as pd

# Load dataset
df = pd.read_csv('Advertising.csv')

# Inspect structure
print(df.head())
print(df.info())

# %%
# Drop the redundant index column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Confirm clean data and check for missing values
print(df.isnull().sum())

# %% [markdown]
# Step 2: Exploratory Data Analysis (EDA)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a Pearson correlation matrix
print(df.corr())

# Plot 1: Heatmap of correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Plot 2: Scatter plots showing trends
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
features = ['TV', 'Radio', 'Newspaper']
for i, col in enumerate(features):
    sns.regplot(x=df[col], y=df['Sales'], ax=axes[i], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    axes[i].set_title(f'{col} budget vs Sales')
plt.tight_layout()
plt.savefig('features_vs_sales.png')
plt.close()

# %% [markdown]
# Step 3: Splitting the Dataset

# %%
from sklearn.model_selection import train_test_split

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Step 4: Machine Learning Model Building
# Model 1: Linear Regression
# Model 2: Random Forest Regressor

# %%
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Initialize and train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# %% [markdown]
# Step 5: Evaluating Regression Performance

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def print_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")

print_metrics(y_test, y_pred_lr, "Linear Regression")
print_metrics(y_test, y_pred_rf, "Random Forest")

# %% [markdown]
# Step 6: Saving the Trained Model

# %%
import joblib

# Export model to file
joblib.dump(rf_model, 'sales_prediction_rf_model.pkl')
print("Sales prediction model saved successfully!")


