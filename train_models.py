import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("\nTraining Heart Disease Model...")
heart = pd.read_csv("heart.csv")

# No encoding needed — all values are numeric

X_heart = heart.drop('condition', axis=1)
y_heart = heart['condition']

X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train, y_train)

print("✅ Heart Model Accuracy:", accuracy_score(y_test, heart_model.predict(X_test)))
pickle.dump(heart_model, open('heart_model.pkl', 'wb'))

# -----------------
# Diabetes Model
# -----------------
print("\nTraining Diabetes Model...")
diabetes = pd.read_csv("diabetes.csv")

# Define features and target
X_dia = diabetes.drop('Outcome', axis=1)
y_dia = diabetes['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_dia, y_dia, test_size=0.2, random_state=42)

# Train model
diabetes_model = RandomForestClassifier(random_state=42)
diabetes_model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, diabetes_model.predict(X_test))
print(f"✅ Diabetes Model Accuracy: {accuracy:.4f}")

# Save model
pickle.dump(diabetes_model, open('diabetes_model.pkl', 'wb'))
