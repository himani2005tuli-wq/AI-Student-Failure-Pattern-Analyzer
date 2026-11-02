# AI Student Failure Pattern Analyzer

**Subtitle:** Analyzing Academic and Behavioral Factors for Student Performance Prediction & Early Risk Detection  
**Author:** Himani Tuli (USN: `25MSDSR074`)  
**Programme:** M.Sc. Data Science and Analytics, JAIN (Deemed-to-be University)  
**Supervisor / Guide:** Dr. M A Ghouse Basha  

---

## 📌 Project Overview
This project presents a machine learning system designed to analyze academic, demographic, lifestyle, and behavioural factors to predict postgraduate students' Master's CGPA and deliver tailored, actionable recommendations for early academic support.

### Key Objectives
1. **Predict Master's CGPA:** Train and benchmark 6 regression algorithms with leak-free preprocessing to estimate academic performance on a 0.0–10.0 CGPA scale.
2. **Personalised Recommendation Engine:** Decoupled heuristic prescriptive engine providing prioritized, explained academic guidance based on documented behavioural thresholds (attendance, study hours, sleep, stress, and procrastination).
3. **What-If Intervention Simulator:** Interactive tool allowing students and advisors to simulate hypothetical lifestyle improvements and examine predicted performance differences ($\Delta = \text{Hypothetical} - \text{Baseline}$).
4. **Batch Analysis & Target Audit:** Transparent batch processing pipeline auditing ground-truth target validity across cohorts.

---

## 🔬 Dataset & Ground-Truth Target Audit
- **Raw Survey Data:** 118 student responses across 29 academic and psychometric survey items (`118 records.csv`).
- **Target Verification:**
  - **Semester 1 Cohort (9 records):** Excluded from supervised training because first-semester students had not completed university Master's end-term exams and recorded prior undergraduate Bachelor's graduation marks.
  - **Zero-Placeholder Records (2 records):** Excluded due to unrecorded entries (`0.00`).
  - **Valid Master's CGPA Targets (107 records):** Semesters 2–4 respondents with verified Master's CGPA (Scale: 6.00 to 10.00; Mean: 8.29).
  - *All 118 records are retained for batch inference with explicit audit flags.*

---

## 📊 Model Benchmark & Evaluation

Models were evaluated using **5-Fold Cross-Validation on the training set (85 records)** and evaluated on a held-out test set (22 records, 20%):

| Algorithm | 5-Fold CV MAE (Train) | 5-Fold CV RMSE (Train) | 5-Fold CV $R^2$ (Train) | Held-Out Test MAE | Held-Out Test RMSE | Held-Out Test $R^2$ | Selection Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Extra Trees Regressor** | **0.6086** | **0.7772** | **+0.0718** | **0.6223** | **0.7877** | **-0.0436** | **Champion Model** |
| Random Forest Regressor | 0.6173 | 0.7873 | +0.0529 | 0.5676 | 0.7500 | +0.0540 | Runner-up |
| Gradient Boosting Regressor | 0.6355 | 0.8144 | -0.0217 | 0.5452 | 0.7508 | +0.0519 | Baseline Boosting |
| Decision Tree Regressor | 0.6437 | 0.8115 | -0.0506 | 0.7467 | 0.8697 | -0.2720 | Overfitting baseline |
| Ridge Regression | 0.6594 | 0.8749 | -0.2324 | 0.6513 | 0.7962 | -0.0661 | Linear regularized |
| Linear Regression | 0.7232 | 0.9343 | -0.4620 | 0.6605 | 0.8040 | -0.0873 | Multicollinear |

### Key Predictors (Extra Trees Gini Importance)
1. **12th Marks:** 23.2%
2. **10th Marks:** 21.7%
3. **Academic Procrastination Score:** 9.1%
4. **Gender (Categorical One-Hot):** 8.8% / 7.6%
5. **Daily Study Hours:** 7.3%
6. **Class Attendance:** 5.1%
7. **Average Sleep Duration:** 4.5%
8. **Perceived Stress Score:** 2.8%

---

## 🗂️ Project Repository Structure

```text
├── 118 records.csv                  # Official survey dataset
├── app.py                           # Full Streamlit interactive dashboard (Tabs 1-4)
├── train_pipeline.py                # End-to-end model training, CV & serialization pipeline
├── test_suite.py                    # 11-point automated verification test suite
├── finalcode .ipynb                 # Comprehensive project Jupyter Notebook
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore configuration
├── Himani_Synopsis.pdf              # Academic project synopsis
├── Himani_Tuli_AI_Risk_Detection... # Academic final project report
├── qrcode_docs.google.com.png       # Survey instrument QR code
└── outputs/                         # Serialized models, metadata & visualizations
    ├── best_pipeline.joblib         # Serialized Extra Trees model pipeline
    ├── benchmark_metrics.joblib     # Cross-validation & test comparison metrics
    ├── feature_importances.joblib   # Gini feature importance data
    ├── training_metadata.joblib     # Model parameters & feature metadata
    └── plots/                       # Architecture, EDA & performance plots
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed. Clone this repository and install dependencies:
```bash
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
pip install -r requirements.txt
```

### 2. Retrain the Pipeline
To re-run data ingestion, target verification, 5-fold cross-validation, and artifact generation:
```bash
python train_pipeline.py
```

### 3. Run Automated Verification Tests
To run the automated 11-step regression and validation test suite:
```bash
python test_suite.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## ⚠️ Advisory & Ethical Notice
The recommendations generated by this system are heuristic educational suggestions designed to promote positive study habits and self-regulation. They do **not** constitute clinical psychological diagnoses, nor do they guarantee academic performance outcomes. Risk categories are rule-based advisory classifications.
