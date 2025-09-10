import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Career Prediction 🎯📊", layout="wide", initial_sidebar_state="expanded")

MODELS_DIR = Path("models")
clf_path = MODELS_DIR / "placement_clf.pkl"
cgpa_path = MODELS_DIR / "cgpa_reg.pkl"
pkg_path = MODELS_DIR / "package_reg.pkl"

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    models['clf'] = joblib.load(clf_path)
    models['cgpa'] = joblib.load(cgpa_path)
    models['pkg'] = joblib.load(pkg_path)
    return models

# ============================
# Sidebar Dataset Insights
# ============================
st.sidebar.title("Dataset & Insights 📈")
if Path("student_career_data.csv").exists():
    df = pd.read_csv("student_career_data.csv")
    st.sidebar.metric("Rows in dataset", len(df))
    st.sidebar.metric("Avg CGPA", round(df['Current_CGPA'].mean(), 2))
    st.sidebar.metric("% placed", f"{round(df['Placed'].mean() * 100, 2)}%")

    if st.sidebar.checkbox("Show charts in sidebar", value=True):
        fig1, ax1 = plt.subplots()
        ax1.hist(df['Current_CGPA'].dropna(), bins=20)
        ax1.set_title("CGPA distribution")
        ax1.set_xlabel("CGPA")
        st.sidebar.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.hist(df.loc[df['Placed'] == 1, 'Expected_Package'], bins=20)
        ax2.set_title("Package (placed students)")
        st.sidebar.pyplot(fig2)
else:
    st.sidebar.info("No dataset found. Run generate_data.py & train_models.py first to enable dataset insights.")

# ============================
# Main Title
# ============================
st.title("Student Career Prediction System 🎯💼📈")
st.markdown("Enter student details below — the app will predict placement eligibility, next semester CGPA, and expected package. It will also give actionable personalized recommendations.")

# ============================
# Input Form
# ============================
with st.form("student_form"):
    c1, c2, c3 = st.columns([2,2,1])
    name = c1.text_input("Name")
    roll = c2.text_input("Roll No")
    branch = c3.selectbox("Branch", ['CSE','IT','ECE','ME','CE','EE'])

    c4, c5, c6 = st.columns(3)
    semester = c4.number_input("Semester", min_value=1, max_value=8, value=6)
    current_cgpa = c5.number_input("Current CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.01, format="%.2f")
    prev_cgpa = c6.number_input("Previous CGPA", min_value=0.0, max_value=10.0, value=6.8, step=0.01, format="%.2f")

    c7, c8, c9 = st.columns(3)
    attendance = c7.slider("Attendance %", 0, 100, 85)
    backlogs = c8.number_input("Backlogs", min_value=0, max_value=10, value=0)
    arrears_cleared = c9.number_input("Arrears Cleared", min_value=0, max_value=10, value=0)

    st.markdown("### Projects / Experience")
    p1, p2, p3, p4 = st.columns(4)
    projects_count = p1.number_input("Projects Count", min_value=0, max_value=20, value=1)
    internships_count = p2.number_input("Internship Count", min_value=0, max_value=10, value=0)
    hackathons = p3.number_input("Hackathons", min_value=0, max_value=10, value=0)
    research_work = p4.selectbox("Research Work", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    st.markdown("### Technical Skills (0-100)")
    t1, t2, t3, t4 = st.columns(4)
    python_s = t1.slider("Python", 0, 100, 65)
    sql_s = t2.slider("SQL", 0, 100, 60)
    ml_s = t3.slider("Machine Learning", 0, 100, 50)
    data_s = t4.slider("Data Analysis (Excel/PowerBI/Tableau)", 0, 100, 55)

    t5, t6, t7 = st.columns(3)
    web_s = t5.slider("Web Dev", 0, 100, 40)
    dsa_s = t6.slider("DSA", 0, 100, 55)
    cloud_s = t7.slider("Cloud (AWS/Azure/GCP)", 0, 100, 35)

    st.markdown("### Soft Skills (0-100)")
    s1, s2, s3, s4 = st.columns(4)
    communication = s1.slider("Communication", 0, 100, 65)
    teamwork = s2.slider("Teamwork", 0, 100, 70)
    problem_solving = s3.slider("Problem Solving", 0, 100, 68)
    leadership = s4.slider("Leadership", 0, 100, 30)

    cert_count = st.number_input("Certifications Count", min_value=0, max_value=50, value=1)
    cert_type = st.selectbox("Cert Type", ['None','Course','Industry','Hackathon'])
    companies_applied = st.number_input("Companies Applied", min_value=0, max_value=200, value=3)
    shortlisted = st.number_input("Shortlisted Count", min_value=0, max_value=100, value=0)
    aptitude_score = st.slider("Aptitude Score (0-100)", 0, 100, 60)
    coding_score = st.slider("Coding Score (0-100)", 0, 100, 65)
    mock_interview_score = st.slider("Mock Interview Score (0-100)", 0, 100, 60)

    clubs = st.number_input("Clubs participated", min_value=0, max_value=10, value=0)
    sports = st.number_input("Sports involvement", min_value=0, max_value=10, value=0)
    lead_role = st.selectbox("Leadership Role", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    confidence = st.slider("Confidence (0-100)", 0, 100, 65)
    stress_handling = st.slider("Stress Handling (0-100)", 0, 100, 55)

    submitted = st.form_submit_button("Predict 🔮")

# ============================
# Predictions
# ============================
if submitted:
    if not (clf_path.exists() and cgpa_path.exists() and pkg_path.exists()):
        st.error("Model files not found in models/. Run train_models.py first.")
    else:
        models = load_models()
        input_df = pd.DataFrame([{
            'Branch': branch,
            'Semester': semester,
            'Current_CGPA': current_cgpa,
            'Previous_CGPA': prev_cgpa,
            'Attendance_%': attendance,
            'Backlogs': backlogs,
            'Arrears_Cleared': arrears_cleared,
            'Strongest_Subject': 'Algorithms',
            'Weakest_Subject': 'Maths',
            'Projects_Count': projects_count,
            'Internships_Count': internships_count,
            'Hackathons': hackathons,
            'Research_Work': research_work,
            'Python': python_s,
            'SQL': sql_s,
            'ML': ml_s,
            'Data_Analysis': data_s,
            'Web_Dev': web_s,
            'DSA': dsa_s,
            'Cloud': cloud_s,
            'Communication': communication,
            'Teamwork': teamwork,
            'Problem_Solving': problem_solving,
            'Leadership': leadership,
            'Cert_Count': cert_count,
            'Cert_Type': cert_type,
            'Companies_Applied': companies_applied,
            'Shortlisted': shortlisted,
            'Aptitude_Score': aptitude_score,
            'Coding_Score': coding_score,
            'Mock_Interview_Score': mock_interview_score,
            'Clubs': clubs,
            'Sports': sports,
            'Lead_Role': lead_role,
            'Confidence': confidence,
            'Stress_Handling': stress_handling
        }])

        placed_proba = models['clf'].predict_proba(input_df)[:,1][0]
        placed_pred = models['clf'].predict(input_df)[0]
        next_cgpa_pred = models['cgpa'].predict(input_df)[0]
        package_pred = models['pkg'].predict(input_df)[0]

        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Placement Eligibility", "Yes" if placed_pred==1 else "No", delta=f"{round(placed_proba*100,2)}% prob")
            st.progress(float(placed_proba))

        with col2:
            st.metric("Predicted Next CGPA", f"{round(next_cgpa_pred,2)} / 10")

        with col3:
            if placed_proba > 0.45:
                st.metric("Expected Package (LPA)", f"{round(package_pred,2)} LPA")
            else:
                st.metric("Expected Package (LPA)", "N/A")

        # Recommendations
        st.markdown("### Personalized Recommendations 💡")
        recs = []
        if current_cgpa < 6.5:
            recs.append("Focus on core subjects: allocate 2-3 hours daily to weak subjects and use concept-building resources.")
        else:
            recs.append("Keep a steady CGPA: continue project work and targeted higher-grade assignments.")

        if coding_score < 60 or dsa_s < 60:
            recs.append("Improve coding & DSA: practice on LeetCode/GFG (2 problems/day).")
        else:
            recs.append("Maintain coding practice with timed mock tests.")

        if communication < 60 or mock_interview_score < 60:
            recs.append("Work on communication: join mock interviews, Toastmasters, or peer groups.")
        else:
            recs.append("Polish behavioral answers for interviews.")

        if projects_count < 2:
            recs.append("Build 1-2 portfolio projects and deploy them.")
        if internships_count == 0:
            recs.append("Apply to internships: startups & micro-internships are good starts.")

        if confidence < 50:
            recs.append("Confidence building: do 1-minute daily talks & present projects to peers.")

        for r in recs:
            st.write("•", r)

        st.markdown("### Top 3 Actions (Quick Wins) 🔥")
        for tip in recs[:3]:
            st.markdown(f"**• {tip}**")
