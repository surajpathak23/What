import os
import streamlit as st
import pandas as pd
import joblib

# Try to load models, if not available then train them
def ensure_models():
    if not (os.path.exists("models/placement_clf.pkl") and 
            os.path.exists("models/cgpa_reg.pkl") and 
            os.path.exists("models/package_reg.pkl")):
        st.warning("⚠️ Models not found! Training models now... Please wait ⏳")
        import train_models  # directly run training script
        train_models.main()  # call main() to train

# Call ensure models
ensure_models()

# Now load the models
placement_model = joblib.load("models/placement_clf.pkl")
cgpa_model = joblib.load("models/cgpa_reg.pkl")
package_model = joblib.load("models/package_reg.pkl")

# Load dataset
df = pd.read_csv("student_career_data.csv")

st.title("🎓 Student Career Prediction App")
st.sidebar.success("Use the menu to predict outcomes.")

# Sidebar metrics
st.sidebar.metric("📈 Total Students", len(df))
st.sidebar.metric("✅ % Placed", f"{round(df['placed'].mean() * 100, 2)}%")
st.sidebar.metric("🎯 Avg Package", f"{round(df['expected_package'].mean(), 2)} LPA")

# Input form
st.header("🔮 Enter New Student Details for Prediction")
with st.form("student_form"):
    semester = st.number_input("Semester", 1, 8, 5)
    current_cgpa = st.number_input("Current CGPA", 0.0, 10.0, 7.0)
    prev_cgpa = st.number_input("Previous CGPA", 0.0, 10.0, 7.0)
    attendance = st.number_input("Attendance %", 0.0, 100.0, 80.0)
    projects_count = st.number_input("Projects Done", 0, 10, 2)
    internships_count = st.number_input("Internships", 0, 5, 1)
    aptitude_score = st.number_input("Aptitude Score", 0, 100, 60)
    coding_score = st.number_input("Coding Score", 0, 100, 70)
    submitted = st.form_submit_button("Predict")

if submitted:
    new_data = pd.DataFrame([{
        "semester": semester,
        "current_cgpa": current_cgpa,
        "prev_cgpa": prev_cgpa,
        "attendance": attendance,
        "projects_count": projects_count,
        "internships_count": internships_count,
        "aptitude_score": aptitude_score,
        "coding_score": coding_score
    }])

    placement_pred = placement_model.predict(new_data)[0]
    cgpa_pred = cgpa_model.predict(new_data)[0]
    package_pred = package_model.predict(new_data)[0]

    st.subheader("📊 Prediction Results")
    st.write("✅ Placement Prediction:", "Placed" if placement_pred == 1 else "Not Placed")
    st.write("📚 Next Semester CGPA Prediction:", round(cgpa_pred, 2))
    st.write("💰 Expected Package:", f"{round(package_pred, 2)} LPA")
