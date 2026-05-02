import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
NUM_RECORDS = 20000  # change to 10000–50000
np.random.seed(42)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def generate_region():
    return np.random.choice(['Rural', 'Suburban', 'Urban'], p=[0.3, 0.3, 0.4])

def generate_learning_trend():
    return np.random.choice(['Increasing', 'Stable', 'Decreasing'], p=[0.4, 0.3, 0.3])

def assign_performance(score, consistency, engagement):
    combined = (0.7 * score) + (35 * consistency) + (0.2 * engagement)

    # Moderate noise
    combined += np.random.normal(0, 5)

    if combined > 95:
        return "High"
    elif combined > 75:
        return "Medium"
    else:
        return "Low"
    

# -----------------------------
# DATA GENERATION
# -----------------------------

data = []

for i in range(NUM_RECORDS):
    student_id = i + 1
    region = generate_region()
    
    time_spent = round(np.random.uniform(1, 15), 1) # hours per week
    modules_completed = np.random.randint(1, 10)      
    quiz_score = round(np.random.uniform(30, 100), 2)
    assignment_timeliness = round(np.random.uniform(40, 100), 2)
    interaction_level = np.random.randint(10, 150)

     # 🔀 ADD OUTLIERS (10% of students)
    if np.random.rand() < 0.03:
        # Contradictory pattern
        quiz_score = np.random.uniform(30, 60)          # low score
        interaction_level = np.random.randint(80, 150)  # high interaction
    
    # Consistency based on randomness (since timestamps removed)
    consistency_index = round(np.random.uniform(0.1, 1.0), 2)

    #NOISE
    quiz_score += np.random.normal(0, 2)
    interaction_level += np.random.normal(0, 3)
    consistency_index = min(max(consistency_index + np.random.normal(0, 0.03), 0), 1)

    quiz_score = min(max(quiz_score, 0), 100)
    interaction_level = max(interaction_level, 0)
    
    # Updated engagement score (no login frequency now)
    engagement_score = round((interaction_level + time_spent * 5) / 2, 2)
    
    learning_trend = generate_learning_trend()

    
    performance_label = assign_performance(quiz_score, consistency_index, engagement_score)


    data.append([
        student_id, region, time_spent, modules_completed, quiz_score,
        assignment_timeliness, interaction_level, consistency_index,
        engagement_score, learning_trend, performance_label
    ])

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

columns = [
    "student_id", "region", "time_spent", "modules_completed",
    "quiz_score", "assignment_timeliness", "interaction_level",
    "consistency_index", "engagement_score",
    "learning_trend" , "performance_label"
]

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# SAVE DATASET
# -----------------------------

df.to_csv("synthetic_student_data.csv", index=False)

print("✅ Dataset generated successfully!")
print(df.head())