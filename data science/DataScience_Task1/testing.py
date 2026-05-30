# %%
import joblib
import pandas as pd

# 1. Load the saved model back into memory
loaded_model = joblib.load('iris_classification_model.pkl')
print("Model loaded perfectly!")

# 2. Define a clean function for classifying new flower measurements
def identify_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    # Construct a DataFrame matching the exact feature order and column names from training
    new_flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], 
        columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    )
    
    # Predict the species
    prediction = loaded_model.predict(new_flower)[0]
    
    # Optional: Get the model's confidence/probability for this prediction
    probabilities = loaded_model.predict_proba(new_flower)[0]
    confidence = max(probabilities) * 100
    
    return prediction, confidence

# 3. Test your reloaded model with new field measurements
# Scenario: A flower with a short, wide sepal and tiny petal (likely Setosa)
species, confidence = identify_iris_species(sepal_length=4.9, sepal_width=3.1, petal_length=1.5, petal_width=0.1)

print(f"\nNew Flower Measurements: 4.9cm x 3.1cm sepal, 1.5cm x 0.1cm petal")
print(f"Model Prediction: {species} ({confidence:.2f}% confidence)")


