import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
df = pd.read_csv("synthetic_student_data.csv")

print("✅ Dataset Loaded")
print(df.head())

# -----------------------------
# 2. BASIC INFO
# -----------------------------
print("\n🔍 Data Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())

# -----------------------------
# 3. MISSING VALUES CHECK
# -----------------------------
print("\n❗ Missing Values:")
print(df.isnull().sum())

# -----------------------------
# 🚀 STEP 4: EDA (DO FIRST)
# -----------------------------

# 📊 1. Quiz Score Distribution
plt.figure()
df["quiz_score"].hist()
plt.title("Quiz Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# 📊 2. Performance Category Distribution
plt.figure()
df["performance_label"].value_counts().plot(kind='bar')
plt.title("Performance Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# 📊 3. Time Spent vs Performance
plt.figure()
sns.boxplot(x="performance_label", y="time_spent", data=df)
plt.title("Time Spent vs Performance")
plt.show()

# 📊 4. Interaction Level vs Performance
plt.figure()
sns.boxplot(x="performance_label", y="interaction_level", data=df)
plt.title("Interaction vs Performance")
plt.show()

# 📊 5. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# 🧹 STEP 3: PREPROCESSING (AFTER EDA)
# -----------------------------

# Drop missing values if any
df = df.dropna()

# Encode categorical features
df = pd.get_dummies(df, columns=["region", "learning_trend"], drop_first=True)

# Separate features and target
X = df.drop("performance_label", axis=1)
y = df["performance_label"]

# Encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print("\n🎯 Target Encoding:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n✅ Preprocessing Completed!")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)