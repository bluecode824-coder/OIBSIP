# %%
import joblib

# 1. Load the vectorizer and model back into memory
loaded_tfidf = joblib.load('tfidf_vectorizer.pkl')
loaded_model = joblib.load('spam_detector_model.pkl')

print("Model and Vectorizer loaded perfectly!")

# 2. Create a clean function for new predictions
def classify_message(text_message):
    # Transform the raw text using the loaded vectorizer
    transformed_text = loaded_tfidf.transform([text_message])
    
    # Predict using the loaded model
    prediction = loaded_model.predict(transformed_text)[0]
    
    return "⚠️ SPAM ALERT!" if prediction == 1 else "✅ Safe (Ham)"

# 3. Test your reloaded model on brand new emails
new_email_1 = "Urgent! Your account has been compromised. Click here to reset your password immediately."
new_email_2 = "Hey mom, I left my keys at your house. Can I come pick them up later?"

print(f"\nEmail 1: {new_email_1}")
print(f"Prediction: {classify_message(new_email_1)}")

print(f"\nEmail 2: {new_email_2}")
print(f"Prediction: {classify_message(new_email_2)}")


