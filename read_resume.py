from pypdf import PdfReader

reader = PdfReader("resume.pdf")

text = ""

for page in reader.pages:
    text = text + page.extract_text()

print(text)



skills = [
    "Python",
    "SQL",
    "Java",
    "Pandas",
    "NumPy",
    "Power BI",
    "Machine Learning",
    "Excel",
    "HTML",
    "CSS",
    "JavaScript"
]

detected_skills = []

for skill in skills:
    if skill.lower() in text.lower():
        detected_skills.append(skill)

print("\nDetected Skills:")
for skill in detected_skills:
    print("-", skill)



    import pandas as pd

jobs = pd.read_csv("jobs.csv")

print("\nAvailable Jobs:")
print(jobs)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume_text = text

job_texts = jobs["Required Skills"].tolist()

vectorizer = TfidfVectorizer()

all_text = [resume_text] + job_texts
vectors = vectorizer.fit_transform(all_text)

similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

jobs["Match Score"] = similarity_scores * 100

jobs = jobs.sort_values(by="Match Score", ascending=False)

print("\nTop 3 Job Recommendations:")

top_jobs = jobs.head(3)

for index, row in top_jobs.iterrows():
    print(row["Job Role"], "-", round(row["Match Score"], 2), "%")


    print("\nSkill Gap Analysis:")

best_job = jobs.iloc[0]

required_skills = best_job["Required Skills"].split(",")

resume_skills_lower = [skill.lower() for skill in detected_skills]

missing_skills = []

for skill in required_skills:
    skill = skill.strip()

    if skill.lower() not in resume_skills_lower:
        missing_skills.append(skill)

print("Recommended Role:", best_job["Job Role"])

print("Missing Skills:")

if missing_skills:
    for skill in missing_skills:
        print("-", skill)
else:
    print("No major skill gap found!")