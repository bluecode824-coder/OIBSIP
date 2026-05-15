import pandas as pd

# Load dataset
df = pd.read_csv("iris.csv")

# Show first 5 rows
print(df.head())

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='Species')
# plt.show()


# 11. Feature Selection

# We separate:

# Inputs → X
# Output → y
X = df.drop('Species', axis=1)
y = df['Species']





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Training the Model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)


# Make Predictions
predictions = model.predict(X_test)

print(predictions)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)