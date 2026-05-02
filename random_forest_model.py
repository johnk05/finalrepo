import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("synthetic_student_data.csv")
df = pd.get_dummies(df, columns=["region", "learning_trend"], drop_first=True)

X = df.drop("performance_label", axis=1)
y = df["performance_label"]

from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report

print("🌲 RANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns

# Create DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Random Forest - Confusion Matrix")
plt.show()

import joblib

joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Random Forest model saved!")