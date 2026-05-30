# %% [markdown]
# Step 1: Data Inspection & Cleaning

# %%
import pandas as pd

# Load the dataset with correct character encoding
df = pd.read_csv("spam.csv", encoding='latin-1')

# Inspect the first few rows and summary information
print(df.head())
print(df.info())

# %%
# Drop unnecessary columns
df = df.dropna(how='any', axis=1)

# Rename columns for clarity
df.columns = ['target', 'text']

# Map the target text labels to binary integers: ham -> 0, spam -> 1
df['label'] = df['target'].map({'ham': 0, 'spam': 1})

print(df.head())

# %% [markdown]
# Step 2: Exploratory Data Analysis (EDA)

# %%
class_counts = df['target'].value_counts()
print(class_counts)

# %% [markdown]
# Feature Engineering: Analyzing Message Lengths
# Does the length of a message help distinguish between a normal email and a spam email?

# %%

import matplotlib.pyplot as plt

# Calculate text lengths
df['length'] = df['text'].apply(len)

# Plot 1: Class Distribution Bar Chart
plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'salmon'])
plt.title('Distribution of Spam and Ham Messages')
plt.xlabel('Type')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# Plot 2: Message Length Histogram
plt.hist(df[df['target'] == 'ham']['length'], bins=50, alpha=0.5, label='Ham', color='blue')
plt.hist(df[df['target'] == 'spam']['length'], bins=50, alpha=0.5, label='Spam', color='red')
plt.title('Message Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('length_distribution.png')
plt.close()

# %% [markdown]
# Step 3: Splitting the Dataset

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

# %% [markdown]
# Step 4: Text Feature Extraction (TF-IDF)
# Machine learning algorithms cannot process raw strings directly. We need to convert our text into numbers. We use TF-IDF (Term Frequency - Inverse Document Frequency) Vectorization for this task.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer and remove common English stop words
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)

# Learn vocabulary from training text and transform it into features
X_train_vec = tfidf.fit_transform(X_train)

# Transform testing text using the learned vocabulary
X_test_vec = tfidf.transform(X_test)

# %% [markdown]
# Step 5: Building and Training Machine Learning Models

# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Train Multinomial Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

# 2. Train Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

# Evaluate Baseline Accuracies
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# %% [markdown]
# Step 6: Detailed Performance Evaluation

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Generate Classification Report for Multinomial Naive Bayes
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Generate and save a Confusion Matrix heatmap
cm = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('confusion_matrix.png')
plt.close()

# %% [markdown]
# Step 7: Testing with Custom Inputs

# %%
def predict_spam_or_ham(message):
    # 1. Convert text to numerical features using our trained TF-IDF instance
    transformed_msg = tfidf.transform([message])
    
    # 2. Run prediction using the model
    prediction = nb_model.predict(transformed_msg)[0]
    
    # 3. Interpret prediction
    if prediction == 1:
        return "⚠️ SPAM ALERT!"
    else:
        return "✅ Safe (Ham)"

# --- Real-world Tests ---

msg_1 = "Hey, are you free tonight? Let's grab dinner around 7 PM."
print(f"Message: '{msg_1}'\nResult: {predict_spam_or_ham(msg_1)}\n")

msg_2 = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341."
print(f"Message: '{msg_2}'\nResult: {predict_spam_or_ham(msg_2)}\n")

# %%
import joblib

# Save the trained Naive Bayes model to a file
joblib.dump(nb_model, 'spam_detector_model.pkl')

# Save the TF-IDF vectorizer to a file
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and Vectorizer saved successfully!")


