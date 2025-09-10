# generate_data.py
import numpy as np
import pandas as pd
import random
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

N = 2000

branches = ['CSE', 'IT', 'ECE', 'ME', 'CE', 'EE']
subjects = ['Algorithms', 'DBMS', 'AI', 'Networks', 'OS', 'Maths', 'Statistics']

def rand_skill():
    # skill score 0-100
    return np.clip(int(np.random.normal(65, 18)), 0, 100)

rows = []
for i in range(N):
    name = fake.first_name()
    roll = f"{2020 + random.randint(0,5)}{random.randint(100,999)}"
    branch = random.choice(branches)
    semester = random.randint(1,8)
    cgpa_current = round(np.clip(np.random.normal(7.0, 1.1), 3.0, 10.0), 2)
    prev_cgpa = round(np.clip(cgpa_current - np.random.normal(0.1, 0.3), 3.0, 10.0), 2)
    attendance = int(np.clip(np.random.normal(85, 10), 50, 100))
    backlogs = np.random.choice([0,0,0,1,2,3], p=[0.6,0.1,0.1,0.1,0.06,0.04])
    arrears_cleared = np.random.randint(0, backlogs+1)
    strongest = random.choice(subjects)
    weakest = random.choice([s for s in subjects if s != strongest])
    projects = np.random.poisson(1.2)
    internships = np.random.choice([0,0,1,1,2], p=[0.5,0.3,0.12,0.05,0.03])
    hackathons = np.random.poisson(0.6)
    research = np.random.choice([0,0,1], p=[0.8,0.15,0.05])
    # technical skills 0-100
    python_s = rand_skill()
    sql_s = rand_skill()
    ml_s = rand_skill()
    data_s = rand_skill()
    web_s = rand_skill()
    dsa_s = rand_skill()
    cloud_s = rand_skill()
    # soft skills
    comm = int(np.clip(np.random.normal(70,15), 30, 100))
    team = int(np.clip(np.random.normal(75,12), 30, 100))
    prob = int(np.clip(np.random.normal(72,14), 30, 100))
    lead = int(np.clip(np.random.normal(60,20), 10, 100))
    cert_count = np.random.poisson(1.5)
    cert_type = np.random.choice(['None','Course','Industry','Hackathon'], p=[0.3,0.4,0.2,0.1])
    companies_applied = np.random.poisson(3)
    shortlisted = max(0, np.random.binomial(companies_applied, 0.15))
    aptitude = int(np.clip(np.random.normal(60,18), 10, 100))
    coding_score = int(np.clip(np.random.normal(65,18), 5, 100))
    mock_interview = int(np.clip(np.random.normal(60,20), 5, 100))
    clubs = np.random.choice([0,1,2,3], p=[0.4,0.3,0.2,0.1])
    sports = np.random.choice([0,1,2], p=[0.7,0.25,0.05])
    leadership_role = np.random.choice([0,1], p=[0.85,0.15])
    confidence = int(np.clip(np.random.normal(68,15), 20, 100))
    stress = int(np.clip(np.random.normal(55,18), 10, 100))

    # simple heuristic for placement eligibility probability and package:
    skill_mean = np.mean([python_s, sql_s, ml_s, dsa_s, data_s, coding_score])
    soft_mean = np.mean([comm, team, prob, lead, mock_interview])
    place_prob = (0.35*skill_mean + 0.25*soft_mean + 0.25*cgpa_current*10 + 0.15*aptitude)/100
    placed = np.random.rand() < np.clip(place_prob, 0.02, 0.98)
    # next CGPA around current +/- small noise influenced by attendance and projects
    next_cgpa = cgpa_current + np.random.normal((attendance-75)/500 + (projects-1)*0.08, 0.4)
    next_cgpa = round(np.clip(next_cgpa, 3.0, 10.0), 2)
    # expected package (LPA)
    base = 2.5 + (skill_mean/100)*5 + (cgpa_current-6)/4*2
    if placed:
        package = round(np.clip(np.random.normal(base + shortlisted*0.7, 1.5), 0.5, 50.0),2)
    else:
        package = 0.0

    rows.append({
        'name': name,
        'roll': roll,
        'branch': branch,
        'semester': semester,
        'current_cgpa': cgpa_current,
        'prev_cgpa': prev_cgpa,
        'attendance': attendance,
        'backlogs': backlogs,
        'arrears_cleared': arrears_cleared,
        'strongest_subject': strongest,
        'weakest_subject': weakest,
        'projects_count': projects,
        'internships_count': internships,
        'hackathons': hackathons,
        'research_work': research,
        'python': python_s,
        'sql': sql_s,
        'ml': ml_s,
        'data_analysis': data_s,
        'web_dev': web_s,
        'dsa': dsa_s,
        'cloud': cloud_s,
        'communication': comm,
        'teamwork': team,
        'problem_solving': prob,
        'leadership': lead,
        'cert_count': cert_count,
        'cert_type': cert_type,
        'companies_applied': companies_applied,
        'shortlisted': shortlisted,
        'aptitude_score': aptitude,
        'coding_score': coding_score,
        'mock_interview_score': mock_interview,
        'clubs': clubs,
        'sports': sports,
        'lead_role': leadership_role,
        'confidence': confidence,
        'stress_handling': stress,
        'placed': int(placed),
        'next_cgpa': next_cgpa,
        'expected_package': package
    })

df = pd.DataFrame(rows)
# introduce some missingness and duplicates
for col in ['python','coding_score','communication','current_cgpa']:
    df.loc[df.sample(frac=0.02, random_state=1).index, col] = np.nan
# duplicates
df = pd.concat([df, df.sample(10, random_state=2)], ignore_index=True)

df.to_csv('student_career_data.csv', index=False)
print("Wrote student_career_data.csv with", len(df), "rows")
