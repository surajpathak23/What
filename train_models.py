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
import joblib

def main():
    DATA = 'student_career_data.csv'
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(DATA)

    # Drop identifiers
    df = df.drop(columns=['name','roll'])

    # Targets
    y_clf = df['placed']
    y_cgpa = df['next_cgpa']
    y_package = df['expected_package']
    X = df.drop(columns=['placed','next_cgpa','expected_package'])

    numeric_features = [c for c in X.select_dtypes(include=['int64','float64']).columns]
    categorical_features = [c for c in X.select_dtypes(include=['object']).columns]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])

    # Split
    X_train, X_test, y_clf_train, y_clf_test, y_cgpa_train, y_cgpa_test, y_pkg_train, y_pkg_test = train_test_split(
        X, y_clf, y_cgpa, y_package, test_size=0.2, random_state=42
    )

    # Models
    clf_pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf_pipeline.fit(X_train, y_clf_train)
    joblib.dump(clf_pipeline, 'models/placement_clf.pkl')

    cgpa_pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    cgpa_pipeline.fit(X_train, y_cgpa_train)
    joblib.dump(cgpa_pipeline, 'models/cgpa_reg.pkl')

    pkg_pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pkg_pipeline.fit(X_train, y_pkg_train)
    joblib.dump(pkg_pipeline, 'models/package_reg.pkl')

    print("✅ All models trained and saved!")

if __name__ == "__main__":
    main()
