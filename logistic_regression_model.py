import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("synthetic_student_data.csv")

# Encode categorical features
df = pd.get_dummies(df, columns=["region", "learning_trend"], drop_first=True)

# Split features and target
X = df.drop("performance_label", axis=1)
y = df["performance_label"]

# -----------------------------
# PREPROCESSING
# -----------------------------
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# LOGISTIC REGRESSION MODEL
# -----------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
from sklearn.metrics import accuracy_score, classification_report

print("📈 LOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.show()

import joblib

joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Logistic Regression model saved!")