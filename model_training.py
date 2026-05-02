import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("synthetic_student_data.csv")

# Encode categorical features
df = pd.get_dummies(df, columns=["region", "learning_trend"], drop_first=True)

# Split features and target
X = df.drop("performance_label", axis=1)
y = df["performance_label"]

# Encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# IMPORT MODELS
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# TRAIN MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []

print("\n🚀 MODEL PERFORMANCE:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append([name, acc, prec, rec, f1])

    print(f"{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)

# -----------------------------
# SHOW COMPARISON TABLE
# -----------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

print("\n📊 MODEL COMPARISON:")
print(results_df.sort_values(by="Accuracy", ascending=False))

# -----------------------------
# 📊 MODEL ACCURACY BAR GRAPH
# ----------------------------

# Extract model names and accuracy from results_df
models = results_df["Model"]
accuracy = results_df["Accuracy"]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()

# Identify top 2 models
top_models = results_df.sort_values(by="Accuracy", ascending=False).head(2)
print("\n🏆 TOP 2 MODELS:")
print(top_models)