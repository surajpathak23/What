# train_models.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import joblib

# ========================
# Load Dataset
# ========================
DATA = 'student_career_data.csv'
os.makedirs('models', exist_ok=True)

df = pd.read_csv(DATA)

# Drop identifiers (if present)
for col in ['name', 'roll']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Define target columns
y_clf = df['placed']
y_cgpa = df['next_cgpa']
y_package = df['expected_package']

X = df.drop(columns=['placed', 'next_cgpa', 'expected_package'])

# ========================
# Feature Lists
# ========================
numeric_features = [
    'semester','current_cgpa','prev_cgpa','attendance','backlogs','arrears_cleared',
    'Projects_Count','Internship_Count','Hackathons','Research_Work','Python','SQL',
    'ML','Data_Analysis','Web_Dev','DSA','Cloud','Communication','Teamwork',
    'Problem_Solving','Leadership','Certifications_Count','Companies_Applied','Shortlisted',
    'Aptitude_Score','Coding_Score','Mock_Interview_Score','Clubs','Sports',
    'Leadership_Role','Confidence','Stress_Handling'
]

categorical_features = ['branch','Strongest_Subject','Weakest_Subject','Cert_Type']

# Keep only columns that actually exist
numeric_features = [f for f in numeric_features if f in X.columns]
categorical_features = [f for f in categorical_features if f in X.columns]

# ========================
# Preprocessing Pipelines
# ========================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
], remainder='drop')

# ========================
# Train / Test Split
# ========================
X_train, X_test, y_clf_train, y_clf_test, y_cgpa_train, y_cgpa_test, y_pkg_train, y_pkg_test = train_test_split(
    X, y_clf, y_cgpa, y_package, test_size=0.2, random_state=42
)

# ========================
# 1) Placement Classifier
# ========================
clf_pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
])

clf_pipeline.fit(X_train, y_clf_train)
preds = clf_pipeline.predict(X_test)
proba = clf_pipeline.predict_proba(X_test)[:, 1]
print("Placement Classifier Accuracy:", accuracy_score(y_clf_test, preds))
print("Placement Classifier ROC AUC:", roc_auc_score(y_clf_test, proba))

joblib.dump(clf_pipeline, 'models/placement_clf.pkl')
print("✅ Saved models/placement_clf.pkl")

# ========================
# 2) Next Semester CGPA Regressor
# ========================
cgpa_pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
])

cgpa_pipeline.fit(X_train, y_cgpa_train)
cgpa_preds = cgpa_pipeline.predict(X_test)
print("CGPA reg RMSE:", mean_squared_error(y_cgpa_test, cgpa_preds, squared=False))
print("CGPA reg R2:", r2_score(y_cgpa_test, cgpa_preds))

joblib.dump(cgpa_pipeline, 'models/cgpa_reg.pkl')
print("✅ Saved models/cgpa_reg.pkl")

# ========================
# 3) Package Regressor
# ========================
pkg_pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

pkg_pipeline.fit(X_train, y_pkg_train)
pkg_preds = pkg_pipeline.predict(X_test)
print("Package reg RMSE:", mean_squared_error(y_pkg_test, pkg_preds, squared=False))
print("Package reg R2:", r2_score(y_pkg_test, pkg_preds))

joblib.dump(pkg_pipeline, 'models/package_reg.pkl')
print("✅ Saved models/package_reg.pkl")
