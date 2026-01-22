# ===============================
# Wine Cultivar Model Development
# ===============================

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target

# 2. Feature selection (6 features)
selected_features = [
    'alcohol',
    'malic_acid',
    'alcalinity_of_ash',
    'magnesium',
    'flavanoids',
    'color_intensity'
]

X = df[selected_features]
y = df['cultivar']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature scaling (MANDATORY)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model training (SVM)
model = SVC(kernel='rbf', probability=True)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Save model and scaler
joblib.dump((model, scaler), "wine_cultivar_model.pkl")

print("Model saved successfully.")
