"""
Machine Learning Pipeline & Model Training Script: AI Student Failure Pattern Analyzer
Project: AI Student Failure Pattern Analyzer - Analyzing Academic and Behavioral Factors for Early Risk Detection

Objective:
Predict postgraduate / Master's cohort academic performance (CGPA) using academic foundation markers
(10th, 12th marks), lifestyle variables, and psychometric behavioral indices (perceived stress and academic procrastination scales).
"""

import os
import sys
import zipfile
import joblib

# Ensure utf-8 output encoding across terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure output directories exist for plots and artifacts
os.makedirs("outputs/plots", exist_ok=True)

# ==============================================================================
# STAGE 1 & 2: DATA COLLECTION, INGESTION & TARGET VERIFICATION
# ==============================================================================
print("\n" + "=" * 78)
print("STAGE 1 & 2: INGESTION, AUDIT & TARGET VARIABLE VERIFICATION")
print("=" * 78)

DATA_PATH = "118 records.csv"

if zipfile.is_zipfile(DATA_PATH):
    with zipfile.ZipFile(DATA_PATH, "r") as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f, encoding="utf-8-sig")
else:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print(f"[+] Loaded raw survey dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Standardize column headers to lowercase snake_case
df.columns = (
    df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
)

# Schema mapping:
# The survey question 'Graduation Marks ' in the Google Form was completed by postgraduate students.
# In Indian higher education, 'Graduation' refers to undergraduate/college degree performance.
# We map it to 'masters_cgpa' as the academic benchmark target representing student CGPA,
# while explicitly documenting the distribution and filtering non-response placeholders.
rename_map = {
    "programme_name_e_g_m_sc_data_science_and_analytics": "programme",
    "current_semester": "semester",
    "10th_marks": "marks_10th",
    "12th_marks": "marks_12th",
    "graduation_marks": "masters_cgpa",
    "average_class_attendance_of_a_subject_per_month": "attendance",
    "study_hour_per_day": "study_hours",
    "sleep_hours": "sleep_hours",
    "what_is_your_current_stress_level": "stress_level",
    "felt_nervous_and_stressed": "stress_q1",
    "unable_to_control_important_things_in_life": "stress_q2",
    "felt_difficulties_piling_up": "stress_q3",
    "felt_confident_handling_problems": "stress_q4",
    "felt_things_were_going_your_way": "stress_q5",
    "i_delay_starting_my_study_tasks": "proc_q1",
    "i_postpone_assignments_until_the_last_moment": "proc_q2",
    "i_avoid_difficult_subjects": "proc_q3",
    "i_study_only_before_exams": "proc_q4",
    "i_get_distracted_while_studying": "proc_q5",
    "i_plan_to_study_but_don_t_follow_the_plan": "proc_q6",
    "i_can_understand_difficult_topics": "proc_q7",
    "i_complete_my_study_goals": "proc_q8",
    "i_can_manage_my_study_time_well": "proc_q9",
    "i_feel_confident_in_exams": "proc_q10"
}
df.rename(columns=rename_map, inplace=True)

# Numerical type conversions
for col in ["age", "marks_10th", "marks_12th", "masters_cgpa"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Audit target scale, missing placeholders, and semester academic validity
total_raw = len(df)
zero_cgpa_count = int(((df["masters_cgpa"].isna()) | (df["masters_cgpa"] <= 0)).sum())
sem1_count = int((df["semester"] == 1).sum())

print(f"[!] Target Audit: Total raw survey records = {total_raw}")
print(f"    - Missing / Zero-placeholder targets : {zero_cgpa_count} records (Rows where CGPA == 0.0)")
print(f"    - Confirmed wrong academic outcome  : {sem1_count} records (Semester 1 respondents who entered Bachelor's graduation marks)")
print(f"    - Total records excluded from training: {zero_cgpa_count + sem1_count} records")

# Exclude records with missing targets or wrong academic outcome from supervised training
training_eligible_mask = (df["masters_cgpa"] > 0) & (df["semester"] > 1)
df_training = df[training_eligible_mask].reset_index(drop=True)

print(f"[+] Valid Master's CGPA Records Retained for Training & Evaluation: {len(df_training)}")
print(f"[+] Valid Master's CGPA Scale: Min = {df_training['masters_cgpa'].min():.2f}, Max = {df_training['masters_cgpa'].max():.2f}, Mean = {df_training['masters_cgpa'].mean():.2f}")

# Assign df to df_training for pipeline modeling
df = df_training

# Midpoint mappings for categorical survey ranges
def convert_study(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    if "0-2" in x: return 1.0
    if "2-4" in x: return 3.0
    if "4-6" in x: return 5.0
    if "more than 6" in x or "6+" in x: return 7.0
    try: return float(x)
    except: return np.nan

def convert_sleep(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    if "<5" in x or "less than 5" in x: return 4.0
    if "5-6" in x: return 5.5
    if "6-8" in x: return 7.0
    if "8+" in x or "more than 8" in x: return 9.0
    try: return float(x)
    except: return np.nan

def convert_attendance(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    if "30" in x and "50" in x: return 40.0
    if "50" in x and "75" in x: return 62.5
    if "75" in x: return 85.0
    try: return float(x)
    except: return np.nan

def clean_programme(p):
    if pd.isna(p): return "Other"
    p = str(p).strip().lower()
    if any(k in p for k in ["data science", "datascience", "dsa", "ai", "aiml", "artificial intelligence", "machine learning"]):
        return "M.Sc Data Science & Analytics"
    elif "mca" in p or "computer application" in p:
        return "MCA"
    elif "biotech" in p:
        return "M.Sc Biotechnology"
    elif "mba" in p or "management" in p:
        return "MBA"
    else:
        return "Other"

df["study_hours"] = df["study_hours"].apply(convert_study)
df["sleep_hours"] = df["sleep_hours"].apply(convert_sleep)
df["attendance"] = df["attendance"].apply(convert_attendance)
df["programme"] = df["programme"].apply(clean_programme)

# Clean gender as categorical text (Female / Male / Other)
df["gender"] = df["gender"].astype(str).str.strip().str.capitalize()
df["gender"] = df["gender"].replace({"Other": "Other", "Male": "Male", "Female": "Female"}).fillna("Other")

# ==============================================================================
# STAGE 3: BEHAVIORAL FEATURE ENGINEERING (PSYCHOMETRICS)
# ==============================================================================
print("\n" + "=" * 78)
print("STAGE 3: PSYCHOMETRIC COMPOSITE FEATURE ENGINEERING")
print("=" * 78)

likert_freq = {"never": 1, "occasionally": 2, "sometimes": 3, "often": 4, "always": 5}
likert_agree = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}

stress_questions = ["stress_q1", "stress_q2", "stress_q3", "stress_q4", "stress_q5"]
for col in stress_questions:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).map(likert_freq)

# Reverse-code positive coping/control items (Q4 & Q5)
df["stress_q4"] = 6 - df["stress_q4"]
df["stress_q5"] = 6 - df["stress_q5"]
df["stress_score"] = df[stress_questions].sum(axis=1, min_count=1)

proc_freq = ["proc_q1", "proc_q2", "proc_q3", "proc_q4", "proc_q5", "proc_q6"]
for col in proc_freq:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).map(likert_freq)

proc_agree = ["proc_q7", "proc_q8", "proc_q9", "proc_q10"]
for col in proc_agree:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).map(likert_agree)
        df[col] = 6 - df[col]  # Reverse-code self-regulation & academic confidence

proc_all = [f"proc_q{i}" for i in range(1, 11)]
df["procrastination_score"] = df[proc_all].sum(axis=1, min_count=1)

print(f"[+] Stress Score Summary: mean = {df['stress_score'].mean():.2f}, range = [{df['stress_score'].min()}, {df['stress_score'].max()}]")
print(f"[+] Procrastination Score Summary: mean = {df['procrastination_score'].mean():.2f}, range = [{df['procrastination_score'].min()}, {df['procrastination_score'].max()}]")

# ==============================================================================
# STAGE 4: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 78)
print("STAGE 4: GENERATING SYSTEM VISUALIZATIONS (CORRELATION & DISTRIBUTIONS)")
print("=" * 78)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Target CGPA Distribution
sns.histplot(df["masters_cgpa"].dropna(), kde=True, ax=axes[0], color="#2563EB", bins=12)
axes[0].set_title("Master's CGPA Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Actual Master's CGPA")
axes[0].axvline(df["masters_cgpa"].mean(), color="red", linestyle="--", label=f"Mean: {df['masters_cgpa'].mean():.2f}")
axes[0].legend()

# Plot 2: Correlation Heatmap
corr_cols = ["marks_10th", "marks_12th", "study_hours", "sleep_hours", "attendance", "stress_score", "procrastination_score", "masters_cgpa"]
corr_mat = df[corr_cols].corr()
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1], annot_kws={"size": 8})
axes[1].set_title("Correlation: Academics & Behaviors vs. CGPA", fontsize=12, fontweight="bold")
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Procrastination vs. CGPA with Trendline
sns.regplot(data=df, x="procrastination_score", y="masters_cgpa", ax=axes[2],
            scatter_kws={"color": "#7C3AED", "alpha": 0.7}, line_kws={"color": "#DC2626", "linewidth": 2})
axes[2].set_title("Procrastination Score vs. Master's CGPA", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Procrastination Score (10-50)")
axes[2].set_ylabel("Master's CGPA")

plt.tight_layout()
plt.savefig("outputs/plots/system_eda_overview.png", dpi=300)
plt.close()
print("[+] Saved visualization: outputs/plots/system_eda_overview.png")

# ==============================================================================
# STAGE 5 & 6: PREPROCESSING PIPELINES & MULTI-MODEL BENCHMARKING
# ==============================================================================
print("\n" + "=" * 78)
print("STAGE 5 & 6: PREPROCESSING PIPELINES & MULTI-MODEL BENCHMARKING")
print("=" * 78)

# Strict candidate features list - target is completely excluded to prevent data leakage
features = [
    "study_hours", "sleep_hours", "attendance", "stress_score",
    "procrastination_score", "age", "gender", "marks_10th", "marks_12th", "programme"
]
target = "masters_cgpa"

df_final = df[features + [target]].dropna(subset=[target]).reset_index(drop=True)
X = df_final[features]
y = df_final[target]

# 80/20 train/test split reserved for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"[+] Total Dataset: {len(df_final)} samples | Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

num_cols = ["marks_10th", "marks_12th", "age", "study_hours", "sleep_hours", "attendance", "stress_score", "procrastination_score"]
cat_cols = ["gender", "programme"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.04, max_depth=2, min_samples_leaf=3, random_state=42),
}

results = []
trained_pipelines = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    
    # 5-Fold Cross Validation evaluated strictly on Training Set to select model
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    cv_mae = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"))
    
    # Train pipeline on full training set
    pipe.fit(X_train, y_train)
    
    # Final evaluation on held-out test set
    preds = pipe.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)
    test_rmse = root_mean_squared_error(y_test, preds)
    test_r2 = r2_score(y_test, preds)
    
    results.append({
        "Model": name,
        "CV MAE (Train)": np.mean(cv_mae),
        "CV RMSE (Train)": np.mean(cv_rmse),
        "CV R2 (Train)": np.mean(cv_r2),
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Test R2": test_r2
    })
    trained_pipelines[name] = pipe

# Benchmark DataFrame sorted by Training Cross-Validation MAE
results_df = pd.DataFrame(results).sort_values(by="CV MAE (Train)", ascending=True).reset_index(drop=True)
print("\n" + "=" * 78 + "\nMODEL BENCHMARK RESULTS (5-FOLD TRAINING CV & HELD-OUT TEST EVALUATION)\n" + "=" * 78)
print(results_df.round(4).to_string(index=False))

# Select champion model based on training cross-validation error
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
best_test_mae = results_df.iloc[0]["Test MAE"]
best_test_rmse = results_df.iloc[0]["Test RMSE"]
best_test_r2 = results_df.iloc[0]["Test R2"]
print(f"\n[*] Selected Champion Model via Training Cross-Validation: {best_model_name}")
print(f"    -> Test MAE: {best_test_mae:.4f} | Test RMSE: {best_test_rmse:.4f} | Test R2: {best_test_r2:.4f}")

# Persist trained model artifacts
MODEL_DIR = "outputs"
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_pipeline.joblib"))
joblib.dump(results_df, os.path.join(MODEL_DIR, "benchmark_metrics.joblib"))
print(f"[+] Serialized model pipeline saved to {os.path.join(MODEL_DIR, 'best_pipeline.joblib')}")
print(f"[+] Serialized benchmark metrics saved to {os.path.join(MODEL_DIR, 'benchmark_metrics.joblib')}")

# Save Actual vs Predicted comparison plot on held-out test set
best_preds = best_pipeline.predict(X_test)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, color="#2563EB", edgecolors="k", s=70, alpha=0.8, label="Test Records (Held-Out)")
min_val = min(y_test.min(), best_preds.min()) - 0.2
max_val = max(y_test.max(), best_preds.max()) + 0.2
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 45° Line")
plt.xlabel("Actual Master's CGPA", fontsize=11, fontweight="bold")
plt.ylabel("Predicted Master's CGPA", fontsize=11, fontweight="bold")
plt.title(f"Actual vs. Predicted Master's CGPA\n{best_model_name} (Test Set)", fontsize=12, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/actual_vs_predicted.png", dpi=300)
plt.close()
print("[+] Saved actual vs predicted plot: outputs/plots/actual_vs_predicted.png")

# ==============================================================================
# STAGE 7: DYNAMIC FEATURE IMPORTANCE & EXPLAINABILITY
# ==============================================================================
print("\n" + "=" * 78)
print("STAGE 7: DYNAMIC FEATURE IMPORTANCE EXTRACTION (CHAMPION MODEL)")
print("=" * 78)

champion_estimator = best_pipeline.named_steps["model"]
preprocessor_step = best_pipeline.named_steps["preprocessor"]

# Extract transformed feature names directly from preprocessor
raw_feat_names = list(preprocessor_step.get_feature_names_out())
clean_feat_names = [f.replace("num__", "").replace("cat__", "") for f in raw_feat_names]

if hasattr(champion_estimator, "feature_importances_"):
    feat_imp = pd.Series(champion_estimator.feature_importances_, index=clean_feat_names).sort_values(ascending=False)
    print(f"\nGini Feature Importance Rankings ({best_model_name}):")
    for feat, imp in feat_imp.head(10).items():
        print(f"  - {feat:<30}: {imp * 100:.2f}%")
        
    joblib.dump(feat_imp, os.path.join(MODEL_DIR, "feature_importances.joblib"))
    print(f"[+] Saved feature importance series: {os.path.join(MODEL_DIR, 'feature_importances.joblib')}")

    # Plot Feature Importances
    plt.figure(figsize=(10, 5))
    feat_imp.head(8).plot(kind="barh", color="#10B981")
    plt.title(f"Gini Feature Importance for Master's CGPA Prediction ({best_model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance (Mean Decrease in Impurity)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance.png", dpi=300)
    plt.close()
    print("[+] Saved feature importance plot: outputs/plots/feature_importance.png")
else:
    print("[!] Model does not have native tree feature_importances_; saving benchmark table.")

# Save training metadata dictionary for frontend synchronization
metadata = {
    "champion_model": best_model_name,
    "valid_records_count": len(df_final),
    "train_samples_count": len(X_train),
    "test_samples_count": len(X_test),
    "test_mae": float(best_test_mae),
    "test_rmse": float(best_test_rmse),
    "test_r2": float(best_test_r2),
    "features": features,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "target": target
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "training_metadata.joblib"))
print(f"[+] Saved training metadata to {os.path.join(MODEL_DIR, 'training_metadata.joblib')}")

# ==============================================================================
# STAGE 8: PERSONALISED RECOMMENDATION ENGINE
# ==============================================================================
def generate_recommendations(student_dict):
    """
    Decoupled rule-based personalised recommendation engine.
    Generates actionable, non-clinical recommendations based on behavioral and academic inputs,
    prioritizing urgent institutional compliance and cognitive fatigue risks.
    """
    recs = []
    
    # 1. Attendance compliance (Critical institutional rule)
    att = float(student_dict.get("attendance", 85.0))
    if att < 75.0:
        recs.append({
            "category": "Attendance & Academic Standing",
            "priority": 1,
            "severity": "High",
            "trigger": f"Reported monthly attendance is {att:.1f}% (< 75% institutional requirement).",
            "recommendation": "Prioritize regular lecture attendance, request lecture recordings/notes for missed classes, and coordinate with course mentors."
        })
        
    # 2. Perceived Stress (Scale: 5-25, Threshold: >= 18)
    stress = float(student_dict.get("stress_score", 14.0))
    if stress >= 18.0:
        recs.append({
            "category": "Stress Management & Well-being",
            "priority": 2,
            "severity": "High",
            "trigger": f"Perceived stress score is elevated at {stress:.0f}/25 (Threshold >= 18).",
            "recommendation": "Utilize university academic counseling or wellness cell resources for structured stress coping strategies (non-clinical guidance)."
        })
        
    # 3. Procrastination & Self-Regulation (Scale: 10-50, Threshold: >= 35)
    proc = float(student_dict.get("procrastination_score", 28.0))
    if proc >= 35.0:
        recs.append({
            "category": "Task Management & Procrastination",
            "priority": 3,
            "severity": "Medium",
            "trigger": f"Academic procrastination score is high at {proc:.0f}/50 (Threshold >= 35).",
            "recommendation": "Deconstruct major term projects into smaller micro-milestones with self-imposed weekly completion targets."
        })
        
    # 4. Daily Study Hours (Threshold: < 3.0 hrs/day)
    study = float(student_dict.get("study_hours", 3.0))
    if study < 3.0:
        recs.append({
            "category": "Study Habits & Timetable",
            "priority": 4,
            "severity": "Medium",
            "trigger": f"Daily independent study duration is {study:.1f} hours/day (< 3.0 hours/day).",
            "recommendation": "Establish a consistent daily timetable allocating 3–4 hours for focused, distraction-free study blocks."
        })
        
    # 5. Sleep Routine (Threshold: < 6.0 hrs/night)
    sleep = float(student_dict.get("sleep_hours", 7.0))
    if sleep < 6.0:
        recs.append({
            "category": "Sleep Hygiene & Recovery",
            "priority": 5,
            "severity": "Low",
            "trigger": f"Average sleep duration is {sleep:.1f} hours/night (< 6.0 hours/night).",
            "recommendation": "Maintain a regular sleep schedule aiming for 7–8 hours of restful nocturnal sleep to support memory consolidation."
        })
        
    # Default positive guidance if no flags triggered
    if not recs:
        recs.append({
            "category": "Performance Maintenance",
            "priority": 6,
            "severity": "Informational",
            "trigger": "All academic and behavioral indicators are within healthy benchmark ranges.",
            "recommendation": "Maintain your current structured daily study routine, balanced sleep schedule, and proactive class participation."
        })
        
    recs.sort(key=lambda r: r["priority"])
    return recs

def predict_student_profile(student_dict):
    """Predict CGPA using the trained pipeline and assign advisory risk tier."""
    df_in = pd.DataFrame([student_dict])
    raw_pred = float(best_pipeline.predict(df_in)[0])
    # Clip only to valid scale domain [0.0, 10.0]
    predicted_cgpa = round(float(np.clip(raw_pred, 0.0, 10.0)), 2)
    
    if predicted_cgpa < 7.0:
        tier = "HIGH RISK"
        desc = "Below 7.0 CGPA - Early Academic Mentorship Required"
    elif predicted_cgpa < 8.0:
        tier = "MEDIUM RISK"
        desc = "7.0 to 7.99 CGPA - Targeted Monitoring Advised"
    else:
        tier = "LOW RISK"
        desc = "8.0+ CGPA - Consistent Performance Standing"
        
    recommendations = generate_recommendations(student_dict)
    return {
        "Predicted_CGPA": predicted_cgpa,
        "Risk_Tier": tier,
        "Risk_Description": desc,
        "Recommendations": recommendations
    }

# Demo evaluation
sample_student = {
    "study_hours": 2.0,
    "sleep_hours": 5.0,
    "attendance": 65.0,
    "stress_score": 19,
    "procrastination_score": 38,
    "age": 22,
    "gender": "Female",
    "marks_10th": 7.5,
    "marks_12th": 7.2,
    "programme": "M.Sc Data Science & Analytics"
}
demo_result = predict_student_profile(sample_student)
print("\n" + "=" * 78)
print("DEMO INFERENCE & PRESCRIPTIVE INTERVENTION OUTPUT:")
print(f"Predicted Master's CGPA: {demo_result['Predicted_CGPA']} / 10.0")
print(f"Advisory Risk Tier     : {demo_result['Risk_Tier']} ({demo_result['Risk_Description']})")
print("Personalised Recommendations:")
for rec in demo_result["Recommendations"]:
    print(f"  • [{rec['priority']}] {rec['category']} ({rec['severity']}): {rec['recommendation']}")
    print(f"    Reason: {rec['trigger']}")
print("=" * 78 + "\n")
