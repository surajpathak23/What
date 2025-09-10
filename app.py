import streamlit as st
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("student_career_data.csv")

# Load models (अगर locally train किए हैं और models/ folder में रखा है तो काम करेगा)
try:
    placement_model = pickle.load(open("models/placement_clf.pkl", "rb"))
    cgpa_model = pickle.load(open("models/cgpa_reg.pkl", "rb"))
    package_model = pickle.load(open("models/package_reg.pkl", "rb"))
except:
    placement_model, cgpa_model, package_model = None, None, None

# Streamlit UI
st.set_page_config(page_title="Student Career Prediction", layout="wide")
st.title("🎓 Student Career Prediction App")

# Sidebar Insights
st.sidebar.header("📊 Dataset & Insights")
st.sidebar.metric("Rows in dataset", df.shape[0])
st.sidebar.metric("Avg CGPA", round(df['Current_CGPA'].mean(), 2))
st.sidebar.metric("Avg Attendance %", round(df['Attendance_%'].mean(), 2))
st.sidebar.metric("% Eligible for Placement", f"{round(df['Placement_Eligibility'].mean() * 100, 2)}%")

# Main Section
st.subheader("🔎 Explore Dataset")
st.dataframe(df.head(10))

# Prediction Form
st.subheader("🤖 Make Predictions")

name = st.text_input("Student Name")
roll_no = st.text_input("Roll Number")
cgpa = st.number_input("Current CGPA", 0.0, 10.0, 7.0)
attendance = st.number_input("Attendance %", 0.0, 100.0, 75.0)
projects = st.number_input("No. of Projects", 0, 20, 2)
internships = st.number_input("No. of Internships", 0, 10, 1)
aptitude = st.number_input("Aptitude Test Score (0-100)", 0, 100, 60)
coding = st.number_input("Coding Test Score (0-100)", 0, 100, 50)

if st.button("Predict Career Outcome"):
    features = [[cgpa, attendance, projects, internships, aptitude, coding]]
    
    if placement_model and cgpa_model and package_model:
        placement = placement_model.predict(features)[0]
        next_cgpa = cgpa_model.predict(features)[0]
        package = package_model.predict(features)[0]

        st.success(f"📌 Prediction for {name} ({roll_no}):")
        st.write("✅ Placement Eligibility:", "Yes" if placement == 1 else "No")
        st.write("📈 Predicted Next CGPA:", round(next_cgpa, 2))
        st.write("💰 Expected Package (LPA):", round(package, 2))
    else:
        st.error("⚠️ Models not found! Please ensure .pkl files are in models/ folder.")
