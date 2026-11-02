import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set page configuration with institutional branding
st.set_page_config(
    page_title="AI Student Failure Pattern Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Professional Polish
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .high-risk {
        background-color: #FEF2F2;
        border: 1px solid #F87171;
        border-radius: 10px;
        padding: 18px;
        color: #991B1B;
    }
    .medium-risk {
        background-color: #FFFBEB;
        border: 1px solid #FBBF24;
        border-radius: 10px;
        padding: 18px;
        color: #92400E;
    }
    .low-risk {
        background-color: #ECFDF5;
        border: 1px solid #34D399;
        border-radius: 10px;
        padding: 18px;
        color: #065F46;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%);
        color: white;
        font-weight: 600;
        font-size: 15px;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1E40AF 0%, #1D4ED8 100%);
        color: #F0FDF4;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:28px;">🎓 AI Student Failure Pattern Analyzer</h1>
    <p style="margin:6px 0 0 0; opacity:0.92; font-size:15px;">
        Analyzing Academic & Behavioral Patterns for Early Risk Detection and Performance Recovery
    </p>
</div>
""", unsafe_allow_html=True)

# Load trained pipeline and artifacts dynamically
MODEL_PATH = os.path.join("outputs", "best_pipeline.joblib")
METRIC_PATH = os.path.join("outputs", "benchmark_metrics.joblib")
FEAT_IMP_PATH = os.path.join("outputs", "feature_importances.joblib")
META_PATH = os.path.join("outputs", "training_metadata.joblib")

pipeline = None
benchmark_df = None
feature_importances = None
metadata = {}

if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        pipeline = None
        st.error(f"Error loading trained model pipeline: {e}")

if os.path.exists(METRIC_PATH):
    try:
        benchmark_df = joblib.load(METRIC_PATH)
    except Exception:
        benchmark_df = None

if os.path.exists(FEAT_IMP_PATH):
    try:
        feature_importances = joblib.load(FEAT_IMP_PATH)
    except Exception:
        feature_importances = None

if os.path.exists(META_PATH):
    try:
        metadata = joblib.load(META_PATH)
    except Exception:
        metadata = {}

# Decoupled Recommendation Engine
def generate_recommendations(student_dict):
    """
    Decoupled rule-based personalised recommendation engine.
    Generates actionable, non-clinical recommendations based on behavioral and academic inputs,
    prioritizing urgent institutional compliance and cognitive wellness risks.
    """
    recs = []
    
    # 1. Attendance compliance (Critical institutional rule)
    att = float(student_dict.get("attendance", 85.0))
    if att < 75.0:
        recs.append({
            "category": "Attendance & Standing",
            "priority": 1,
            "severity": "Critical",
            "trigger": f"Reported monthly attendance is {att:.1f}% (Institutional minimum is 75%).",
            "recommendation": "Prioritize class presence immediately. Obtain lecture notes and recordings for missed classes, and schedule a consultation with your course faculty mentor."
        })
        
    # 2. Perceived Stress (Scale: 5-25, Threshold: >= 18)
    stress = float(student_dict.get("stress_score", 14.0))
    if stress >= 18.0:
        recs.append({
            "category": "Stress Management",
            "priority": 2,
            "severity": "High",
            "trigger": f"Perceived stress score is elevated at {stress:.0f}/25 (Advisory threshold >= 18).",
            "recommendation": "Utilize university academic mentoring, wellness cell resources, or counseling workshops for stress-reduction strategies (supportive guidance, non-clinical)."
        })
        
    # 3. Academic Procrastination (Scale: 10-50, Threshold: >= 35)
    proc = float(student_dict.get("procrastination_score", 28.0))
    if proc >= 35.0:
        recs.append({
            "category": "Task Management",
            "priority": 3,
            "severity": "Medium",
            "trigger": f"Procrastination score is high at {proc:.0f}/50 (Advisory threshold >= 35).",
            "recommendation": "Deconstruct large coursework and term projects into modular micro-deliverables with clear, self-imposed weekly completion targets."
        })
        
    # 4. Daily Study Hours (Threshold: < 3.0 hrs/day)
    study = float(student_dict.get("study_hours", 3.0))
    if study < 3.0:
        recs.append({
            "category": "Study Habits",
            "priority": 4,
            "severity": "Medium",
            "trigger": f"Daily independent study duration is {study:.1f} hours/day (< 3.0 hours/day).",
            "recommendation": "Establish a consistent daily study timetable allocating 3–4 hours for focused, distraction-free study blocks."
        })
        
    # 5. Sleep Duration (Threshold: < 6.0 hrs/night)
    sleep = float(student_dict.get("sleep_hours", 7.0))
    if sleep < 6.0:
        recs.append({
            "category": "Sleep Routine",
            "priority": 5,
            "severity": "Low",
            "trigger": f"Average sleep duration is {sleep:.1f} hours/night (< 6.0 hours/night).",
            "recommendation": "Maintain a regular nocturnal sleep routine aiming for 7–8 hours nightly to prevent cognitive fatigue and improve memory consolidation."
        })
        
    # Default positive guidance if no flags triggered
    if not recs:
        recs.append({
            "category": "Maintenance",
            "priority": 6,
            "severity": "Informational",
            "trigger": "All academic and behavioral indicators are within healthy benchmark ranges.",
            "recommendation": "Maintain your current structured daily study routine, balanced sleep schedule, and active class participation."
        })
        
    recs.sort(key=lambda r: r["priority"])
    return recs

# Sidebar - Project Overview & Navigation
with st.sidebar:
    if os.path.exists("qrcode_docs.google.com.png"):
        st.image("qrcode_docs.google.com.png", width=140, caption="Survey Ingestion Link")
    
    st.markdown("### 📌 Objective & Model Summary")
    
    champ_name = metadata.get("champion_model", "Random Forest")
    test_mae = metadata.get("test_mae", 0.5626)
    test_rmse = metadata.get("test_rmse", 0.7479)
    test_r2 = metadata.get("test_r2", 0.2419)
    total_samples = metadata.get("valid_records_count", 116)
    
    st.markdown(f"""
    - **Prediction Target:** Master's Cohort CGPA (Scale: 0.0 – 10.0)
    - **Dataset Scale:** 118 survey respondents ({total_samples} valid non-zero records)
    - **Selected Champion Model:** **{champ_name}**
      - *Test MAE:* **{test_mae:.4f}**
      - *Test RMSE:* **{test_rmse:.4f}**
      - *Test R²:* **{test_r2:.4f}**
    """)
    
    if feature_importances is not None and not feature_importances.empty:
        st.markdown("---")
        st.markdown("### 🔍 Top Predictor Contributors")
        st.caption("Gini Feature Importance (MDI) dynamically extracted from trained champion model:")
        for idx, (feat, val) in enumerate(feature_importances.head(5).items(), 1):
            clean_name = feat.replace("marks_", "").replace("_", " ").title()
            st.markdown(f"**{idx}. {clean_name}:** `{val * 100:.1f}%`")
        st.caption("⚠️ *Reflects model predictive association, not causal proof.*")
        
    st.markdown("---")
    st.markdown("### ⚙️ Quick Presets")
    preset_choice = st.selectbox(
        "Load Student Persona Profile",
        ["Custom Input", "High-Risk Candidate (Illustrative)", "Average Student with High Stress (Illustrative)", "High Achiever (Illustrative)"]
    )
    st.caption("Select an illustrative persona preset to auto-populate the assessment form.")

# Parse Preset Values
default_10th = 8.5
default_12th = 8.2
default_prog = "M.Sc Data Science & Analytics"
default_gender = "Female"
default_age = 22
default_study_idx = 1
default_sleep_idx = 2
default_attend_idx = 2
default_stress = 14
default_proc = 28

if "High-Risk Candidate" in preset_choice:
    default_10th = 6.2
    default_12th = 5.8
    default_prog = "M.Sc Data Science & Analytics"
    default_gender = "Male"
    default_age = 23
    default_study_idx = 0  # 0-2 hours
    default_sleep_idx = 0  # <5 hrs
    default_attend_idx = 0  # 30%-50%
    default_stress = 22
    default_proc = 42
elif "Average Student with High Stress" in preset_choice:
    default_10th = 7.8
    default_12th = 7.5
    default_prog = "M.Sc Data Science & Analytics"
    default_gender = "Female"
    default_age = 22
    default_study_idx = 1  # 2-4 hours
    default_sleep_idx = 1  # 5-6 hrs
    default_attend_idx = 1  # 50%-75%
    default_stress = 21
    default_proc = 36
elif "High Achiever" in preset_choice:
    default_10th = 9.4
    default_12th = 9.2
    default_prog = "M.Sc Data Science & Analytics"
    default_gender = "Female"
    default_age = 22
    default_study_idx = 2  # 4-6 hours
    default_sleep_idx = 2  # 6-8 hrs
    default_attend_idx = 2  # 75% and above
    default_stress = 9
    default_proc = 16

# Main Tabs
tab_eval, tab_simulator, tab_batch, tab_analytics, tab_methodology = st.tabs([
    "🔍 Real-Time Master's Risk Evaluation",
    "⚡ What-If Intervention Simulator",
    "📁 Batch Survey Analysis & Early Warning Alert",
    "📊 Model Performance & EDA",
    "📐 System Methodology & Architecture"
])

# -----------------------------------------------------------------------------
# TAB 1: INDIVIDUAL STUDENT EVALUATION
# -----------------------------------------------------------------------------
with tab_eval:
    st.markdown("#### Enter Student Academic Foundations & Psychometric Survey Indicators")
    st.caption("The system predicts the student's estimated **Master's CGPA** using the trained machine learning pipeline and generates rule-based personalized guidance.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📚 Academic Foundation")
        marks_10th = st.slider("10th Grade Marks (Scale 4.0 - 10.0)", 4.0, 10.0, float(default_10th), 0.1, help="Normalized Secondary School CGPA or Marks/10")
        marks_12th = st.slider("12th Grade Marks (Scale 4.0 - 10.0)", 4.0, 10.0, float(default_12th), 0.1, help="Normalized Higher Secondary CGPA or Marks/10")
        
        prog_options = ["M.Sc Data Science & Analytics", "MCA", "M.Sc Biotechnology", "MBA", "Other"]
        prog_idx = prog_options.index(default_prog) if default_prog in prog_options else 0
        programme = st.selectbox("Postgraduate Programme", prog_options, index=prog_idx)
        
        age = st.number_input("Age", min_value=17, max_value=40, value=int(default_age), step=1)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0 if default_gender == "Female" else (1 if default_gender == "Male" else 2))

    with col2:
        st.markdown("##### ⏳ Study & Lifestyle Habits")
        study_cat = st.selectbox("Daily Study Hours", ["0-2 hours", "2-4 hours", "4-6 hours", "More than 6 hours"], index=default_study_idx)
        study_map = {"0-2 hours": 1.0, "2-4 hours": 3.0, "4-6 hours": 5.0, "More than 6 hours": 7.0}
        study_hours = study_map[study_cat]
        
        sleep_cat = st.selectbox("Average Sleep Duration", ["Less than 5 hrs", "5-6 hrs", "6-8 hrs", "8+ hrs"], index=default_sleep_idx)
        sleep_map = {"Less than 5 hrs": 4.0, "5-6 hrs": 5.5, "6-8 hrs": 7.0, "8+ hrs": 9.0}
        sleep_hours = sleep_map[sleep_cat]
        
        attend_cat = st.selectbox("Average Monthly Attendance", ["30%-50%", "50%-75%", "75% and above"], index=default_attend_idx)
        attend_map = {"30%-50%": 40.0, "50%-75%": 62.5, "75% and above": 85.0}
        attendance = attend_map[attend_cat]

    with col3:
        st.markdown("##### 🧠 Psychometric & Behavioral Scales")
        stress_score = st.slider(
            "Perceived Stress Score (5-25)", 
            min_value=5, max_value=25, value=int(default_stress),
            help="Sum of 5-item Perceived Stress Scale with reverse coding on positive coping items"
        )
        proc_score = st.slider(
            "Composite Procrastination Score (10-50)", 
            min_value=10, max_value=50, value=int(default_proc),
            help="Sum of 10-item Academic Procrastination Scale with reverse coding on self-regulation items"
        )
        st.caption(f"**Current Behavioral Index:** Stress: {stress_score}/25 | Procrastination: {proc_score}/50")

    st.markdown("---")
    
    # Run evaluation automatically on input change and provide refresh button
    st.button("🔄 Recalculate Evaluation", use_container_width=True)
    
    if pipeline is None:
        st.error("🚨 Trained Model Pipeline Not Found! Please run `python train_pipeline.py` to train and save `outputs/best_pipeline.joblib` before running inference.")
    else:
        student_data = {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "attendance": attendance,
            "stress_score": stress_score,
            "procrastination_score": proc_score,
            "age": age,
            "gender": gender,
            "marks_10th": marks_10th,
            "marks_12th": marks_12th,
            "programme": programme
        }
        
        df_in = pd.DataFrame([student_data])
        raw_pred = float(pipeline.predict(df_in)[0])
        # Scale constraints applied strictly to valid CGPA domain [0.0, 10.0]
        pred_cgpa = round(float(np.clip(raw_pred, 0.0, 10.0)), 2)
        
        # Rule-based institutional advisory risk tiers
        if pred_cgpa < 7.0:
            risk_tier = "HIGH RISK"
            risk_class = "high-risk"
            icon = "🚨"
            status_text = "Advisory Tier: Immediate Multi-Disciplinary Master's Academic Mentorship Advised (< 7.0 CGPA)"
        elif pred_cgpa < 8.0:
            risk_tier = "MEDIUM RISK"
            risk_class = "medium-risk"
            icon = "⚠️"
            status_text = "Advisory Tier: Targeted Monitoring, Attendance Verification & Faculty Mentorship Advised (7.0 - 7.99 CGPA)"
        else:
            risk_tier = "LOW RISK"
            risk_class = "low-risk"
            icon = "✅"
            status_text = "Advisory Tier: Consistent Academic Standing (>= 8.0 CGPA)"

        # Display Outcome Card
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="margin:0 0 8px 0;">{icon} Risk Tier: {risk_tier}</h3>
                <h1 style="margin:0; font-size:36px; font-weight:bold;">{pred_cgpa:.2f} / 10.0</h1>
                <p style="margin:8px 0 0 0; font-size:14px;"><strong>Estimated Master's CGPA</strong></p>
                <hr style="margin:12px 0;">
                <p style="margin:0; font-size:13px;">{status_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.subheader("📋 Personalised Academic & Behavioral Recommendations")
            recommendations = generate_recommendations(student_data)
            
            for r in recommendations:
                badge_color = "#DC2626" if r["severity"] in ["Critical", "High"] else ("#D97706" if r["severity"] == "Medium" else "#2563EB")
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border-left: 4px solid {badge_color}; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                    <span style="font-weight: bold; font-size: 14px;">[{r['category']}]</span> - {r['recommendation']}<br>
                    <small style="color: #64748B;"><em>Trigger: {r['trigger']}</em></small>
                </div>
                """, unsafe_allow_html=True)
            st.caption("ℹ️ *Recommendations are rule-based advisory suggestions designed for student support and do not guarantee grade increases.*")

        st.markdown("---")
        # Multi-Dimensional Diagnostics & CGPA Gauge
        st.markdown("#### 🎯 Multi-Dimensional Behavioral & Academic Diagnostics")
        rad_col1, rad_col2 = st.columns(2)
        
        with rad_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_cgpa,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Master's CGPA (0.0-10.0 Scale)", 'font': {'size': 17}},
                delta={'reference': 8.0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0.0, 10.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1E3A8A"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0.0, 7.0], 'color': '#FCA5A5'},
                        {'range': [7.0, 8.0], 'color': '#FDE68A'},
                        {'range': [8.0, 10.0], 'color': '#86EFAC'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 7.0
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with rad_col2:
            norm_prior_acad = ((marks_10th + marks_12th) / 20.0) * 100
            norm_study = (study_hours / 7.0) * 100
            norm_attendance = (attendance / 85.0) * 100
            norm_wellness = ((25.0 - stress_score) / 20.0) * 100  # inverted stress
            norm_discipline = ((50.0 - proc_score) / 40.0) * 100  # inverted procrastination
            
            categories = ['Prior Academics', 'Study Discipline', 'Attendance', 'Emotional Wellness', 'Execution Focus']
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[norm_prior_acad, norm_study, norm_attendance, norm_wellness, norm_discipline],
                theta=categories,
                fill='toself',
                name='Student Profile',
                line_color='#2563EB',
                fillcolor='rgba(37, 99, 235, 0.3)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[80, 70, 85, 75, 75],
                theta=categories,
                name='Benchmark Target',
                line=dict(color='#10B981', dash='dot')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=280,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: WHAT-IF INTERVENTION SIMULATOR
# -----------------------------------------------------------------------------
with tab_simulator:
    st.markdown("### ⚡ What-If Prescriptive Intervention Simulator")
    st.write(
        "Simulate how hypothetical lifestyle changes (e.g. increasing study time, attending classes regularly, or reducing procrastination) "
        "are projected to relate to a student's estimated **Master's CGPA** using the trained machine-learning model."
    )
    st.info("ℹ️ **Methodological Note:** This simulation reflects statistical associations learned by the machine-learning model from the survey cohort. It is an advisory decision-support tool, not causal proof that an intervention will guarantee grade changes.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.markdown("##### 📍 Baseline Student State")
        b_10th = st.number_input("Baseline 10th Marks", 4.0, 10.0, 7.2, 0.1, key="b_10th")
        b_12th = st.number_input("Baseline 12th Marks", 4.0, 10.0, 6.8, 0.1, key="b_12th")
        b_prog = st.selectbox("Baseline Programme", ["M.Sc Data Science & Analytics", "MCA", "M.Sc Biotechnology", "MBA", "Other"], key="b_prog")
        b_gender = st.selectbox("Baseline Gender", ["Female", "Male", "Other"], key="b_gender")
        b_age = st.number_input("Baseline Age", 17, 35, 22, key="b_age")
        b_study = st.select_slider("Baseline Study Hours/Day", [1.0, 3.0, 5.0, 7.0], value=1.0, key="b_study")
        b_sleep = st.select_slider("Baseline Sleep Duration", [4.0, 5.5, 7.0, 9.0], value=4.0, key="b_sleep")
        b_attend = st.select_slider("Baseline Attendance %", [40.0, 62.5, 85.0], value=40.0, key="b_attend")
        b_stress = st.slider("Baseline Stress Score (5-25)", 5, 25, 21, key="b_stress")
        b_proc = st.slider("Baseline Procrastination (10-50)", 10, 50, 42, key="b_proc")

    with sim_col2:
        st.markdown("##### 🛠️ Hypothetical Interventions / Habit Adjustments")
        st.caption("Adjust behavioral parameters to simulate hypothetical post-intervention scenarios:")
        i_study = st.slider("Target Daily Study Hours (e.g. guided study sessions)", 1.0, 7.0, 5.0, 1.0)
        i_sleep = st.slider("Target Sleep Hours (e.g. sleep hygiene routine)", 4.0, 9.0, 7.0, 0.5)
        i_attend = st.slider("Target Attendance % (e.g. mandatory attendance recovery)", 40.0, 85.0, 85.0, 5.0)
        i_stress = st.slider("Target Stress Score (e.g. wellness workshop)", 5, 25, 12, 1)
        i_proc = st.slider("Target Procrastination (e.g. modular micro-milestones)", 10, 50, 22, 1)

    st.markdown("---")
    
    # Simulation calculates dynamically on input adjustment
    st.button("🔄 Update What-If Simulation", use_container_width=True)
    
    if pipeline is None:
        st.error("🚨 Trained Model Pipeline Not Found! Please run `python train_pipeline.py` before running simulations.")
    else:
        base_dict = {
            "study_hours": b_study, "sleep_hours": b_sleep, "attendance": b_attend,
            "stress_score": b_stress, "procrastination_score": b_proc,
            "age": b_age, "gender": b_gender, "marks_10th": b_10th, "marks_12th": b_12th,
            "programme": b_prog
        }
        interv_dict = {
            "study_hours": i_study, "sleep_hours": i_sleep, "attendance": i_attend,
            "stress_score": i_stress, "procrastination_score": i_proc,
            "age": b_age, "gender": b_gender, "marks_10th": b_10th, "marks_12th": b_12th,
            "programme": b_prog
        }
        
        raw_base = float(pipeline.predict(pd.DataFrame([base_dict]))[0])
        raw_interv = float(pipeline.predict(pd.DataFrame([interv_dict]))[0])
        
        # Apply scale constraints strictly to valid range [0.0, 10.0]
        cgpa_base = round(float(np.clip(raw_base, 0.0, 10.0)), 2)
        cgpa_interv = round(float(np.clip(raw_interv, 0.0, 10.0)), 2)
        cgpa_delta = round(cgpa_interv - cgpa_base, 2)
        
        sim_res1, sim_res2, sim_res3 = st.columns(3)
        with sim_res1:
            st.metric("Baseline Master's CGPA", f"{cgpa_base:.2f}")
            st.caption("Baseline Advisory Standing: " + ("🚨 High Risk (<7.0)" if cgpa_base < 7.0 else ("⚠️ Medium Risk (7.0-7.99)" if cgpa_base < 8.0 else "✅ Low Risk (>=8.0)")))
        
        with sim_res2:
            # Correctly format positive, negative, and zero differences
            if cgpa_delta > 0:
                delta_str = f"+{cgpa_delta:.2f} CGPA"
                delta_color = "normal"
            elif cgpa_delta < 0:
                delta_str = f"{cgpa_delta:.2f} CGPA"
                delta_color = "inverse"
            else:
                delta_str = "0.00 CGPA (Unchanged)"
                delta_color = "off"
                
            st.metric("Simulated Post-Intervention CGPA", f"{cgpa_interv:.2f}", delta=delta_str, delta_color=delta_color)
            st.caption("Projected Advisory Standing: " + ("🚨 High Risk (<7.0)" if cgpa_interv < 7.0 else ("⚠️ Medium Risk (7.0-7.99)" if cgpa_interv < 8.0 else "✅ Low Risk (>=8.0)")))
        
        with sim_res3:
            if cgpa_delta > 0:
                if cgpa_base < 7.0 and cgpa_interv >= 7.0:
                    status_msg = "Successfully Elevated to Safe Standing!"
                else:
                    status_msg = f"Positive Projected Association (+{cgpa_delta:.2f} CGPA)"
                st.success(f"**Impact Trajectory:** {status_msg}")
            elif cgpa_delta < 0:
                st.warning(f"**Impact Trajectory:** Negative Projected Association ({cgpa_delta:.2f} CGPA). Worsening habits are correlated with lower predicted performance.")
            else:
                st.info("**Impact Trajectory:** Neutral / No Projected Difference. Feature changes had negligible impact under the model.")

        # Bar chart comparison
        fig_sim = go.Figure(data=[
            go.Bar(name='Baseline State', x=["Master's CGPA"], y=[cgpa_base], marker_color='#EF4444'),
            go.Bar(name='Hypothetical Scenario', x=["Master's CGPA"], y=[cgpa_interv], marker_color='#10B981' if cgpa_delta >= 0 else '#F59E0B')
        ])
        fig_sim.update_layout(barmode='group', yaxis_range=[0.0, 10.0], height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_sim, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: BATCH SURVEY ANALYSIS & INSTITUTIONAL TRIAGE
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📁 Batch Survey Processing & Automated Master's Risk Triage")
    st.write("Process cohort survey records, distinguish actual vs. predicted values, and export institutional early warning reports.")

    uploaded_file = st.file_uploader("Upload CSV Cohort Extract", type=["csv"])
    
    batch_source = None
    if uploaded_file is not None:
        batch_source = pd.read_csv(uploaded_file)
    elif os.path.exists("118 records.csv"):
        st.info("💡 Loaded Active Cohort Database: `118 records.csv`")
        try:
            import zipfile
            if zipfile.is_zipfile("118 records.csv"):
                with zipfile.ZipFile("118 records.csv", "r") as z:
                    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                    with z.open(csv_names[0]) as f:
                        batch_source = pd.read_csv(f, encoding="utf-8-sig")
            else:
                batch_source = pd.read_csv("118 records.csv", encoding="utf-8-sig")
        except Exception as e:
            st.error(f"Error loading survey records: {e}")

    if batch_source is not None:
        df_batch = batch_source.copy()
        df_batch.columns = (
            df_batch.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        )
        
        rename_map = {
            "programme_name_e_g_m_sc_data_science_and_analytics": "programme",
            "10th_marks": "marks_10th", "12th_marks": "marks_12th",
            "graduation_marks": "actual_masters_cgpa",
            "average_class_attendance_of_a_subject_per_month": "attendance",
            "study_hour_per_day": "study_hours", "sleep_hours": "sleep_hours",
            "felt_nervous_and_stressed": "stress_q1", "unable_to_control_important_things_in_life": "stress_q2",
            "felt_difficulties_piling_up": "stress_q3", "felt_confident_handling_problems": "stress_q4",
            "felt_things_were_going_your_way": "stress_q5",
            "i_delay_starting_my_study_tasks": "proc_q1", "i_postpone_assignments_until_the_last_moment": "proc_q2",
            "i_avoid_difficult_subjects": "proc_q3", "i_study_only_before_exams": "proc_q4",
            "i_get_distracted_while_studying": "proc_q5", "i_plan_to_study_but_don_t_follow_the_plan": "proc_q6",
            "i_can_understand_difficult_topics": "proc_q7", "i_complete_my_study_goals": "proc_q8",
            "i_can_manage_my_study_time_well": "proc_q9", "i_feel_confident_in_exams": "proc_q10"
        }
        df_batch.rename(columns=rename_map, inplace=True)
        
        # Mappings using identical preprocessing
        likert_freq = {"never": 1, "occasionally": 2, "sometimes": 3, "often": 4, "always": 5}
        likert_agree = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}
        
        for col in ["stress_q1", "stress_q2", "stress_q3", "stress_q4", "stress_q5"]:
            if col in df_batch.columns:
                df_batch[col] = df_batch[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).map(likert_freq).fillna(3)
        if "stress_q4" in df_batch.columns: df_batch["stress_q4"] = 6 - df_batch["stress_q4"]
        if "stress_q5" in df_batch.columns: df_batch["stress_q5"] = 6 - df_batch["stress_q5"]
        df_batch["stress_score"] = df_batch[["stress_q1", "stress_q2", "stress_q3", "stress_q4", "stress_q5"]].sum(axis=1)

        for col in [f"proc_q{i}" for i in range(1, 7)]:
            if col in df_batch.columns:
                df_batch[col] = df_batch[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).map(likert_freq).fillna(3)
        for col in [f"proc_q{i}" for i in range(7, 11)]:
            if col in df_batch.columns:
                df_batch[col] = df_batch[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).map(likert_agree).fillna(3)
                df_batch[col] = 6 - df_batch[col]
        df_batch["procrastination_score"] = df_batch[[f"proc_q{i}" for i in range(1, 11)]].sum(axis=1)

        def parse_study(x):
            if pd.isna(x): return 3.0
            x = str(x).strip().lower()
            if "0-2" in x: return 1.0
            if "2-4" in x: return 3.0
            if "4-6" in x: return 5.0
            if "more than 6" in x or "6+" in x: return 7.0
            try: return float(x)
            except: return 3.0

        def parse_sleep(x):
            if pd.isna(x): return 7.0
            x = str(x).strip().lower()
            if "<5" in x or "less" in x: return 4.0
            if "5-6" in x: return 5.5
            if "6-8" in x: return 7.0
            if "8+" in x or "more than 8" in x: return 9.0
            try: return float(x)
            except: return 7.0

        def parse_att(x):
            if pd.isna(x): return 85.0
            x = str(x).strip().lower()
            if "30" in x and "50" in x: return 40.0
            if "50" in x and "75" in x: return 62.5
            if "75" in x: return 85.0
            try: return float(x)
            except: return 85.0

        def parse_prog(p):
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

        df_batch["study_hours"] = df_batch["study_hours"].apply(parse_study) if "study_hours" in df_batch.columns else 3.0
        df_batch["sleep_hours"] = df_batch["sleep_hours"].apply(parse_sleep) if "sleep_hours" in df_batch.columns else 7.0
        df_batch["attendance"] = df_batch["attendance"].apply(parse_att) if "attendance" in df_batch.columns else 85.0
        df_batch["programme"] = df_batch["programme"].apply(parse_prog) if "programme" in df_batch.columns else "M.Sc Data Science & Analytics"

        for col in ["age", "marks_10th", "marks_12th"]:
            if col in df_batch.columns:
                df_batch[col] = pd.to_numeric(df_batch[col], errors="coerce")
            else:
                df_batch[col] = 8.0

        # Handle Actual CGPA distinction and semester validity audit
        if "semester" in df_batch.columns:
            df_batch["semester"] = pd.to_numeric(df_batch["semester"], errors="coerce").fillna(2)
        else:
            df_batch["semester"] = 2

        if "actual_masters_cgpa" in df_batch.columns:
            df_batch["actual_masters_cgpa"] = pd.to_numeric(df_batch["actual_masters_cgpa"], errors="coerce")
            
            # Label target validity status for full transparency
            def get_target_status(row):
                cgpa = row.get("actual_masters_cgpa", np.nan)
                sem = row.get("semester", 2)
                if pd.isna(cgpa) or cgpa <= 0:
                    return "Missing / Placeholder (0.0)"
                elif sem == 1:
                    return "Invalid: Semester 1 (Bachelor's Marks)"
                else:
                    return "Valid Master's CGPA"
                    
            df_batch["Target_Audit_Status"] = df_batch.apply(get_target_status, axis=1)
            # Only genuine Master's CGPA (Semesters 2, 3, 4 with positive score) are eligible for model evaluation
            df_batch["evaluation_eligible"] = (df_batch["actual_masters_cgpa"] > 0) & (df_batch["actual_masters_cgpa"] <= 10.0) & (df_batch["semester"] > 1)
        else:
            df_batch["actual_masters_cgpa"] = np.nan
            df_batch["Target_Audit_Status"] = "Not Provided"
            df_batch["evaluation_eligible"] = False

        if "gender" in df_batch.columns:
            df_batch["gender"] = df_batch["gender"].astype(str).str.strip().str.capitalize()
            df_batch["gender"] = df_batch["gender"].replace({"Other": "Other", "Male": "Male", "Female": "Female"}).fillna("Other")
        else:
            df_batch["gender"] = "Other"

        features_needed = ["study_hours", "sleep_hours", "attendance", "stress_score", "procrastination_score", "age", "gender", "marks_10th", "marks_12th", "programme"]
        
        if pipeline is not None:
            batch_preds = pipeline.predict(df_batch[features_needed])
            df_batch["Predicted_Masters_CGPA"] = np.clip(np.round(batch_preds, 2), 0.0, 10.0)
        else:
            st.error("🚨 Trained Model Pipeline Not Found! Batch predictions cannot be generated without loading `outputs/best_pipeline.joblib`.")
            df_batch["Predicted_Masters_CGPA"] = np.nan

        df_batch["Risk_Tier"] = pd.cut(
            df_batch["Predicted_Masters_CGPA"],
            bins=[-0.1, 6.99, 7.99, 10.0],
            labels=["High Risk (<7.0)", "Medium Risk (7.0-7.99)", "Low Risk (>=8.0)"]
        )

        # Overview statistics
        total_recs = len(df_batch)
        eval_recs = int(df_batch["evaluation_eligible"].sum())
        sem1_recs = int((df_batch["semester"] == 1).sum())
        zero_recs = int(((df_batch["actual_masters_cgpa"].isna()) | (df_batch["actual_masters_cgpa"] <= 0)).sum())
        
        st.markdown(f"#### Cohort Processing Summary: {total_recs} Records Ingested")
        st.write(f"- **Eligible for Model Prediction:** {total_recs} records (all student input profiles are preserved for inference)")
        st.write(f"- **Confirmed Valid Master's CGPA for Performance Evaluation:** **{eval_recs} records**")
        st.write(f"- **Excluded from Supervised Evaluation:** **{total_recs - eval_recs} records** ({sem1_recs} Semester 1 bachelor's marks + {zero_recs} zero-placeholders)")
        
        if eval_recs > 0 and pipeline is not None:
            eval_mask = df_batch["evaluation_eligible"]
            y_act = df_batch.loc[eval_mask, "actual_masters_cgpa"]
            y_prd = df_batch.loc[eval_mask, "Predicted_Masters_CGPA"]
            
            c_mae = mean_absolute_error(y_act, y_prd)
            c_rmse = root_mean_squared_error(y_act, y_prd)
            c_r2 = r2_score(y_act, y_prd)
            
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Cohort Evaluation MAE ({eval_recs} Eligible Rows)", f"{c_mae:.4f}")
            m2.metric("Cohort Evaluation RMSE", f"{c_rmse:.4f}")
            m3.metric("Cohort Evaluation R² Score", f"{c_r2:.4f}")

        tier_counts = df_batch["Risk_Tier"].value_counts()
        b_c1, b_c2, b_c3 = st.columns(3)
        b_c1.metric("🚨 High Risk Students", int(tier_counts.get("High Risk (<7.0)", 0)), "Advisory Support Needed")
        b_c2.metric("⚠️ Medium Risk Students", int(tier_counts.get("Medium Risk (7.0-7.99)", 0)), "Targeted Mentorship")
        b_c3.metric("✅ Low Risk Students", int(tier_counts.get("Low Risk (>=8.0)", 0)), "Consistent Standing")

        st.markdown("#### Cohort At-Risk Student Triage Table")
        
        triage_cols = [
            "username" if "username" in df_batch.columns else "programme",
            "semester", "marks_10th", "marks_12th", "attendance", "stress_score", "procrastination_score",
            "actual_masters_cgpa", "Target_Audit_Status", "Predicted_Masters_CGPA", "Risk_Tier"
        ]
        triage_view = df_batch[[c for c in triage_cols if c in df_batch.columns]].copy()
        triage_view.rename(columns={"actual_masters_cgpa": "Actual_CGPA (Observed)"}, inplace=True)
        st.dataframe(triage_view.sort_values(by="Predicted_Masters_CGPA", ascending=True), use_container_width=True)

        # CSV Downloadable Report
        csv_download = triage_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Official Institutional At-Risk Early Warning Report (CSV)",
            data=csv_download,
            file_name="Student_Academic_Early_Risk_Report.csv",
            mime="text/csv",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# TAB 4: MODEL BENCHMARK & EDA
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown("#### 🏆 Multi-Model Benchmark Comparison on Master's CGPA (5-Fold CV & Test Set)")
    st.caption("All 6 candidate regression models were trained with ColumnTransformer pipelines. Model selection was performed via 5-Fold Cross-Validation on the training data; the held-out test set was reserved for unbiased final evaluation.")
    
    if benchmark_df is not None:
        st.dataframe(benchmark_df.style.highlight_min(subset=["CV MAE (Train)", "Test MAE", "Test RMSE"], color="#DCFCE7")
                               .highlight_max(subset=["CV R2 (Train)", "Test R2"], color="#DCFCE7"),
                     use_container_width=True)
                     
        # Interactive Plotly Benchmark Chart
        fig_bm = px.bar(
            benchmark_df,
            x="Model",
            y="Test MAE",
            color="Test MAE",
            title="Algorithm Comparison: Test Mean Absolute Error (Lower is Better)",
            labels={"Test MAE": "Test MAE (CGPA Points)"},
            color_continuous_scale="Blues_r"
        )
        fig_bm.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bm, use_container_width=True)
    else:
        st.warning("Benchmark metrics will appear here after executing `python train_pipeline.py`.")

    st.markdown("---")
    st.markdown("#### 📊 System Visualizations: Actual vs. Predicted & Feature Importance")
    c_img1, c_img2 = st.columns(2)
    
    with c_img1:
        st.markdown("##### 🎯 Actual vs. Predicted CGPA (Held-Out Test Set)")
        if os.path.exists("outputs/plots/actual_vs_predicted.png"):
            st.image("outputs/plots/actual_vs_predicted.png", caption="Actual vs. Predicted Master's CGPA (Held-Out Test Set)", use_container_width=True)
        else:
            st.info("Actual vs. Predicted plot available after running `python train_pipeline.py`.")
            
    with c_img2:
        st.markdown("##### 🌲 Gini Feature Importance (Champion Model)")
        if os.path.exists("outputs/plots/feature_importance.png"):
            st.image("outputs/plots/feature_importance.png", caption="Gini Feature Importance for Master's CGPA", use_container_width=True)
        elif feature_importances is not None and not feature_importances.empty:
            fig_fi = px.bar(
                x=feature_importances.values[:10],
                y=[f.replace("marks_", "").replace("_", " ").title() for f in feature_importances.index[:10]],
                orientation="h",
                labels={"x": "Relative Importance", "y": "Feature"},
                title="Gini Feature Importance (MDI)"
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=350)
            st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📈 System Exploratory Overview & Correlation Matrix")
    if os.path.exists("outputs/plots/system_eda_overview.png"):
        st.image("outputs/plots/system_eda_overview.png", caption="System Exploratory Overview: Target Distribution, Correlation Matrix & Procrastination Slope", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: SYSTEM METHODOLOGY & ARCHITECTURE
# -----------------------------------------------------------------------------
with tab_methodology:
    st.markdown("### 🏛️ Complete System Architecture Flowchart (Stages 1 to 8)")
    if os.path.exists("outputs/plots/architecture_flowchart.png"):
        st.image("outputs/plots/architecture_flowchart.png", caption="End-to-End System Architecture & Methodology Flowchart", use_container_width=True)
    
    st.markdown("""
    #### Architectural Breakdown:
    1. **Stage 1 (Data Ingestion & Cohort Audit):** Ingestion of psychometric Google Form survey records (118 postgraduate students across 29 questions covering academic foundations, lifestyle routines, and validated psychometric items).
    2. **Stage 2 (Target Verification & Preprocessing):** Verified target variable as reported cumulative CGPA (survey column `'Graduation Marks '`, scale 0.0–10.0). Identified and filtered 2 unentered zero-placeholder records (`0.00`), retaining 116 validated records for model development.
    3. **Stage 3 (Psychometric Composite Feature Engineering):** Reverse-coded positive coping items (`stress_q4`, `stress_q5`) to create **Perceived Stress Score (5–25)**; reverse-coded self-regulation items (`proc_q7` to `proc_q10`) to create **Academic Procrastination Score (10–50)**.
    4. **Stage 4 (Exploratory Data Analysis):** Analyzed correlation heatmaps, feature distributions, and multicollinearity patterns.
    5. **Stage 5 (Leakage-Free Pipeline Architecture):** Built Scikit-Learn `ColumnTransformer` wrapping numerical imputation (`median`) and `StandardScaler` for continuous features, alongside categorical imputation (`most_frequent`) and `OneHotEncoder(handle_unknown='ignore')` for `gender` and `programme`. Target variable strictly excluded from features.
    6. **Stage 6 (Multi-Model 5-Fold Cross-Validation & Benchmark):** Trained 6 regression algorithms (Linear Regression, Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting) using 80/20 train/test split. Champion model selected based strictly on 5-fold cross-validation on the training set.
    7. **Stage 7 (Explainability & Feature Importance):** Dynamic extraction of Gini feature importances (Mean Decrease in Impurity) indicating primary predictive contributions from academic foundations (12th Marks, 10th Marks) and behavioral self-regulation (Procrastination, Stress).
    8. **Stage 8 (Decoupled Personalised Prescriptions):** Independent rule-based advisory engine generating actionable, prioritized interventions across attendance, stress, procrastination, study time, and sleep hygiene.
    """)
