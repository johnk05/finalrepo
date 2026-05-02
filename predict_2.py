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
    risk_flags = []

    # -----------------------------
    # PERFORMANCE BASED
    # -----------------------------
    if label == "Low":
        rec.append("Focus on building strong fundamentals and revise core concepts regularly.")
    elif label == "Medium":
        rec.append("You are performing moderately well. Target weak areas to improve further.")
    elif label == "High":
        rec.append("You are performing well. Continue your current strategy and aim for consistency.")

    # -----------------------------
    # CONSISTENCY ANALYSIS
    # -----------------------------
    if row["consistency_index"] < 0.5:
        rec.append("Low consistency detected. Create a fixed daily study schedule.")
        risk_flags.append("Low Consistency Risk")

    elif row["consistency_index"] > 0.8:
        rec.append("Excellent consistency. Maintain this habit for sustained performance.")

    # -----------------------------
    # QUIZ PERFORMANCE
    # -----------------------------
    if row["quiz_score"] < 60:
        rec.append("Low quiz performance. Focus on concept clarity and practice more problems.")
        risk_flags.append("Low Score Risk")

    elif row["quiz_score"] > 85:
        rec.append("Strong quiz performance. You can move to advanced-level questions.")

    # -----------------------------
    # INTERACTION ANALYSIS
    # -----------------------------
    if row["interaction_level"] < 50:
        rec.append("Low interaction detected. Participate more in discussions and group learning.")
    
    elif row["interaction_level"] > 100:
        rec.append("High interaction level. Good engagement with learning environment.")

    # -----------------------------
    # EFFORT vs OUTPUT (SMART INSIGHT)
    # -----------------------------
    if row["time_spent"] > 10 and row["quiz_score"] < 60:
        rec.append("High effort but low performance detected. Improve study strategy and focus on understanding concepts.")

    if row["time_spent"] < 4 and row["quiz_score"] > 80:
        rec.append("High efficiency detected. You learn quickly—consider exploring advanced topics.")

    # -----------------------------
    # LEARNING TREND ANALYSIS
    # -----------------------------
    if row["learning_trend"] == "Decreasing":
        rec.append("Performance trend is declining. Review recent topics and identify weak areas.")
        risk_flags.append("Declining Trend Risk")

    elif row["learning_trend"] == "Increasing":
        rec.append("Positive learning trend. Keep up the good progress.")

    # -----------------------------
    # AT-RISK DETECTION (VERY IMPORTANT)
    # -----------------------------
    if (
        row["consistency_index"] < 0.5 and
        row["quiz_score"] < 60 and
        row["learning_trend"] == "Decreasing"
    ):
        rec.append("⚠️ You are at risk of poor performance. Immediate intervention is recommended.")

    # -----------------------------
    # REMOVE DUPLICATES
    # -----------------------------
    rec = list(set(rec))

    # -----------------------------
    # RETURN OUTPUT
    # -----------------------------

    return {
        "recommendations": rec,
        "risks": list(set(risk_flags))
    }

   
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

        result = get_recommendation(prediction, student_row.iloc[0])

        print("\n💡 Recommendations:")
        for r in result["recommendations"]:
            print("-", r)

        if result["risks"]:
            print("\n⚠️ Risk Factors:")
            for r in result["risks"]:
                print("-", r)