import pandas as pd
import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("synthetic_student_data.csv")

# -----------------------------
# LOAD PREPROCESSING OBJECTS
# -----------------------------
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# LOAD MODEL FUNCTION
# -----------------------------
def load_model(choice):
    if choice == 1:
        return joblib.load("logistic_model.pkl"), "Logistic Regression"
    elif choice == 2:
        return joblib.load("rf_model.pkl"), "Random Forest"
    else:
        print("❌ Invalid choice")
        return None, None

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_student(student_row, model):
    # Drop target column
    data = student_row.drop("performance_label", axis=1)

    # One-hot encoding
    data = pd.get_dummies(data)

    # Align columns
    data = data.reindex(columns=columns, fill_value=0)

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    pred = model.predict(data_scaled)
    label = le.inverse_transform(pred)

    return label[0]

# -----------------------------
# RECOMMENDATION SYSTEM
# -----------------------------
def get_recommendation(label, row):
    rec = []

    if label == "Low":
        rec.append("Improve consistency and study regularly")
        rec.append("Revise basic concepts")

    elif label == "Medium":
        rec.append("Increase practice and engagement")
        rec.append("Work on improving quiz scores")

    elif label == "High":
        rec.append("Maintain current performance")
        rec.append("Explore advanced topics")

    # Feature-based suggestions
    if row["consistency_index"] < 0.5:
        rec.append("Increase study consistency")

    if row["quiz_score"] < 60:
        rec.append("Focus on improving quiz performance")

    if row["interaction_level"] < 50:
        rec.append("Participate more in discussions")

    return rec

# -----------------------------
# MAIN PROGRAM
# -----------------------------
print("\n🔹 Select Model:")
print("1. Logistic Regression")
print("2. Random Forest")

choice = int(input("Enter choice (1/2): "))
model, model_name = load_model(choice)

if model is not None:
    student_id = int(input("\nEnter Student ID: "))

    # Fetch student data
    student_row = df[df["student_id"] == student_id]

    if student_row.empty:
        print("❌ Student ID not found")
    else:
        student_row = student_row.copy()

        print("\n📊 Student Data:")
        print(student_row)

        prediction = predict_student(student_row, model)

        print(f"\n🎯 Model Used: {model_name}")
        print(f"📈 Predicted Performance: {prediction}")

        recommendations = get_recommendation(prediction, student_row.iloc[0])

        print("\n💡 Recommendations:")
        for r in recommendations:
            print("-", r)