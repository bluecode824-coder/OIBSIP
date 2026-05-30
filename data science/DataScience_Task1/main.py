# %% [markdown]
# Step 1: Data Inspection & Cleaning

# %%
import pandas as pd

# Load dataset
df = pd.read_csv('iris.csv')

# Inspect structure
print(df.head())
print(df.info())
print(df['Species'].value_counts())

# %%
# Remove the index column
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# %% [markdown]
# Step 2: Training and Evaluating the Model

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Separate features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Stratified split to keep equal class representation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the Multiclass Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
print(f"Overall Model Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Plot and save the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Iris Classification')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.tight_layout()
plt.savefig('iris_confusion_matrix.png')
plt.close()

# Save the trained model to disk
joblib.dump(model, 'iris_classification_model.pkl')
print("Model saved successfully as 'iris_classification_model.pkl'!")

# %% [markdown]
# Step 3: Production Testing Script

# %%
import os
import joblib
import pandas as pd

def test_iris_predictor():
    model_filename = 'iris_classification_model.pkl'
    
    # 1. Verification Check: Ensure the model file exists
    if not os.path.exists(model_filename):
        print(f" Error: '{model_filename}' not found in the current folder.")
        print("Please execute your training script first to generate this file.")
        return
    
    # 2. Load the trained model into memory
    try:
        model = joblib.load(model_filename)
        print("Saved Iris Classification model loaded successfully!\n")
    except Exception as e:
        print(f" Failed to load the model file. Error: {e}")
        return

    # 3. Create an explicit inference helper function
    def classify_new_flower(sepal_length, sepal_width, petal_length, petal_width):
        # Build DataFrame with the exact feature column names used during training
        feature_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]], 
            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        )
        
        # Predict the category class name string
        predicted_species = model.predict(feature_df)[0]
        
        # Calculate the model's prediction probability/confidence level
        probabilities = model.predict_proba(feature_df)[0]
        max_confidence = max(probabilities) * 100
        
        return predicted_species, max_confidence

    # 4. Run Test Scenarios with Mock Measurements
    print("--- Running Test Predictions ---")
    
    # Test 1: Typical Setosa metrics (Short sepals, wide sepals, small petals)
    species1, conf1 = classify_new_flower(sepal_length=4.8, sepal_width=3.4, petal_length=1.6, petal_width=0.2)
    print(f"Scenario 1: Sepal: 4.8x3.4 cm | Petal: 1.6x0.2 cm")
    print(f"  ↳ Predicted Class: {species1} ({conf1:.1f}% confidence)")

    # Test 2: Typical Versicolor metrics (Mid-range sizes)
    species2, conf2 = classify_new_flower(sepal_length=6.1, sepal_width=2.8, petal_length=4.7, petal_width=1.2)
    print(f"\nScenario 2: Sepal: 6.1x2.8 cm | Petal: 4.7x1.2 cm")
    print(f"  ↳ Predicted Class: {species2} ({conf2:.1f}% confidence)")

    # Test 3: Typical Virginica metrics (Long, wide petals)
    species3, conf3 = classify_new_flower(sepal_length=7.2, sepal_width=3.0, petal_length=6.0, petal_width=2.2)
    print(f"\nScenario 3: Sepal: 7.2x3.0 cm | Petal: 6.0x2.2 cm")
    print(f"  ↳ Predicted Class: {species3} ({conf3:.1f}% confidence)")

if __name__ == '__main__':
    test_iris_predictor()

# %% [markdown]
# Save the Trained Model

# %%
import joblib

# Save your trained model to a file
joblib.dump(model, 'iris_classification_model.pkl')

print("Iris classification model saved successfully!")


