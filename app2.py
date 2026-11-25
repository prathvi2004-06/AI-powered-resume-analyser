import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import hmac
import secrets
import io
from pathlib import Path
from pypdf import PdfReader
try:
    from docx import Document
except ImportError:
    Document = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ResumeIQ | AI Resume Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DESIGN SYSTEM --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#172554; --muted:#64748b; --blue:#4f46e5; --line:#e5eaf4; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f5f7fc; color:#172033; }
.block-container { max-width:1440px; padding-top:1.4rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#101a35; border-right:1px solid #1d2b4c; }
[data-testid="stSidebar"] * { color:#e7ecff; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { background:transparent; padding:9px 12px; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#202d50; }
.brand { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:#fff; letter-spacing:-.5px; }
.brand span { color:#9da7ff; }
.side-sub { color:#9aa8c8; font-size:.82rem; margin-top:-8px; margin-bottom:1.2rem; }
.hero { background:linear-gradient(120deg,#171f48 0%,#3730a3 55%,#6366f1 100%); padding:30px 32px; border-radius:24px; color:white; margin-bottom:22px; box-shadow:0 12px 35px #252b6c1c; }
.hero h1 { color:white; font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-1px; font-size:2rem; margin:0; }
.hero p { color:#e1e7ff; margin:.55rem 0 0; font-size:1rem; }
.eyebrow { text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:1.5px; color:#c7d2fe; margin-bottom:10px; }
.page-title { font-family:'Manrope',sans-serif; font-size:1.65rem; font-weight:800; color:#172554; margin:0 0 4px 0; letter-spacing:-.5px; }
.page-subtitle { color:#64748b; margin-bottom:20px; }
div[data-testid="stMetric"] { background:white; border:1px solid #e5eaf4; border-radius:18px; padding:18px 18px 14px; box-shadow:0 5px 18px #19284b08; }
div[data-testid="stMetricLabel"] { color:#64748b; font-weight:600; }
div[data-testid="stMetricValue"] { color:#172554; font-family:'Manrope',sans-serif; font-weight:800; }
.card { background:white; border:1px solid #e5eaf4; border-radius:18px; padding:20px 22px; margin-bottom:14px; box-shadow:0 5px 18px #19284b07; }
.card h3 { margin:0 0 7px; color:#172554; font-family:'Manrope',sans-serif; font-size:1.08rem; }
.card p { color:#64748b; margin:0; }
.pill { display:inline-block; border-radius:30px; background:#eef2ff; color:#4338ca; padding:5px 10px; margin:4px 4px 4px 0; font-size:.82rem; font-weight:600; }
.pill-green { background:#e7f8ef; color:#147348; }
.pill-red { background:#fff1f2; color:#be123c; }
div[data-testid="stFileUploader"] { background:white; padding:16px; border:1.5px dashed #a5b4fc; border-radius:16px; }
.stButton button, div[data-testid="stFormSubmitButton"] button { border-radius:11px; font-weight:700; min-height:42px; }
div[data-testid="stProgressBar"] > div { border-radius:20px; }
hr { border-color:#e5eaf4; }
.footer { text-align:center; color:#94a3b8; font-size:.82rem; padding:28px 0 8px; }
.auth-wrap { background:white; border:1px solid #e5eaf4; border-radius:24px; padding:28px; box-shadow:0 15px 45px #1b285b0d; }
[data-testid="stSidebar"] button { width:100%; text-align:left; border:1px solid #26365b; background:#172544; color:#eef2ff; margin:3px 0; }
[data-testid="stSidebar"] button:hover { border-color:#818cf8; background:#25345a; color:white; }
.page-title { font-size:1.9rem; padding-top:.2rem; }
[data-testid="stAppViewContainer"] .main .block-container { padding-top:2rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- DATABASE / AUTHENTICATION --------------------
DB_PATH = Path(__file__).with_name("resume_users.db")
JOBS_PATH = Path(__file__).with_name("jobs.csv")


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def initialize_user_database():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resume_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                resume_name TEXT NOT NULL,
                resume_hash TEXT NOT NULL,
                ats_score REAL NOT NULL,
                skills_count INTEGER NOT NULL,
                best_role TEXT,
                match_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_email, resume_hash)
            )
        """)
        conn.commit()


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()
    return digest, salt


def create_user(full_name, email, password):
    digest, salt = hash_password(password)
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                (full_name.strip(), email.strip().lower(), digest, salt),
            )
            conn.commit()
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except sqlite3.Error:
        return False, "Database error. Please try again."


def verify_user(email, password):
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT id, full_name, email, password_hash, salt FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        if not row:
            return None
        user_id, name, saved_email, saved_hash, salt = row
        entered_hash, _ = hash_password(password, salt)
        if hmac.compare_digest(entered_hash, saved_hash):
            return {"id": user_id, "name": name, "email": saved_email}
        return None
    except sqlite3.Error:
        return None


def google_auth_configured():
    """Return True when Streamlit OIDC settings for Google are present."""
    try:
        auth = st.secrets.get("auth", {})
        return bool(
            auth.get("redirect_uri")
            and auth.get("cookie_secret")
            and auth.get("client_id")
            and auth.get("client_secret")
            and auth.get("server_metadata_url")
        )
    except Exception:
        return False


def get_google_user():
    """Read the signed-in OIDC identity, if Streamlit supports OIDC here."""
    try:
        if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
            email = str(getattr(st.user, "email", "") or "").strip().lower()
            name = str(getattr(st.user, "name", "") or email.split("@")[0] or "Google user")
            if email:
                return {"id": f"google:{email}", "name": name, "email": email, "auth_provider": "google"}
    except Exception:
        pass
    return None


initialize_user_database()
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = get_google_user()
# On OAuth callback/rerun, populate the app session from the verified Google identity.
if st.session_state.logged_in_user is None:
    st.session_state.logged_in_user = get_google_user()

# -------------------- LOGIN / SIGN-UP SCREEN --------------------
if st.session_state.logged_in_user is None:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("""
        <div class="hero" style="min-height:390px;display:flex;flex-direction:column;justify-content:center;">
          <div class="eyebrow">YOUR CAREER, WITH MORE CLARITY</div>
          <h1 style="font-size:2.4rem;">Make your resume<br>work smarter.</h1>
          <p>Analyze your skills, explore relevant roles, and build a focused improvement plan with AI-assisted insights.</p>
          <br>
          <div class="card" style="background:#ffffff12;border-color:#ffffff25;color:white;">
            <h3 style="color:white;">✦ One workspace for your job search</h3>
            <p style="color:#e1e7ff;">Resume score · Job matching · Skill gaps · Improvement plan</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Built with Python · Streamlit · SQLite · Scikit-learn</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="page-title">Welcome to ResumeIQ</div><div class="page-subtitle">Log in or create your free project account.</div>', unsafe_allow_html=True)
        login_tab, signup_tab = st.tabs(["Login", "Create account"])
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email address", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login to dashboard", use_container_width=True)
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter your email and password.")
                else:
                    user = verify_user(email, password)
                    if user:
                        st.session_state.logged_in_user = user
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")
        with signup_tab:
            with st.form("signup_form", clear_on_submit=True):
                name = st.text_input("Full name", key="signup_name")
                new_email = st.text_input("Email address", key="signup_email")
                new_password = st.text_input("Create password (8+ characters)", type="password", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                submitted_signup = st.form_submit_button("Create account", use_container_width=True)
            if submitted_signup:
                if not all([name.strip(), new_email.strip(), new_password, confirm]):
                    st.error("Please fill in all fields.")
                elif "@" not in new_email or "." not in new_email.rsplit("@", 1)[-1]:
                    st.error("Enter a valid email address.")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters.")
                elif new_password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = create_user(name, new_email, new_password)
                    st.success(message) if ok else st.error(message)
        st.markdown("<div style='text-align:center;margin:18px 0 10px;color:#64748b;'>OR</div>", unsafe_allow_html=True)
        if google_auth_configured() and hasattr(st, "login"):
            if st.button("Continue with Google", use_container_width=True, type="secondary"):
                # Credentials are configured under the top-level [auth] section
                # in .streamlit/secrets.toml, so call st.login() without a provider.
                st.login()
        else:
            st.button("Continue with Google", use_container_width=True, disabled=True, help="Set up Google OAuth in .streamlit/secrets.toml first.")
            st.caption("Google sign-in needs OAuth credentials. See setup instructions in the downloaded guide or ask your project supervisor for permission to configure a Google Cloud OAuth client.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -------------------- SIDEBAR NAVIGATION --------------------
# Button navigation avoids the radio/select circle and applies page changes before rerun.
NAV_ITEMS = [
    ("🏠", "Dashboard"), ("📄", "Resume Analyzer"), ("🎯", "Job Matches"),
    ("🧩", "Skill Gap Plan"), ("✨", "Resume Improvement"),
    ("📝", "Resume Builder"), ("🕘", "Resume History"), ("👤", "My Account")
]
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"
if "pending_navigation" in st.session_state:
    st.session_state.active_page = st.session_state.pop("pending_navigation").replace("🏠  ", "").replace("📄  ", "").replace("🎯  ", "").replace("🧩  ", "").replace("👤  ", "").strip()

def persist_resume_upload():
    current_upload = st.session_state.get("resume_uploader_widget")
    if current_upload is not None:
        st.session_state.saved_resume_bytes = current_upload.getvalue()
        st.session_state.saved_resume_name = current_upload.name

with st.sidebar:
    st.markdown('<div class="brand">Resume<span>IQ.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">AI Resume & Career Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("WORKSPACE")
    for icon, label in NAV_ITEMS:
        button_label = f"{icon}  {label}"
        if st.button(button_label, key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label
            st.rerun()
    page = st.session_state.active_page
    st.markdown("---")
    st.caption("SIGNED IN AS")
    st.markdown(f"**{st.session_state.logged_in_user['name']}**")
    st.caption(st.session_state.logged_in_user["email"])
    if st.button("↪ Log out", use_container_width=True, key="logout_button"):
        was_google_login = st.session_state.logged_in_user.get("auth_provider") == "google"
        st.session_state.logged_in_user = None
        for key in ["resume_uploader_widget", "saved_resume_bytes", "saved_resume_name", "active_page"]:
            st.session_state.pop(key, None)
        if was_google_login and hasattr(st, "logout"):
            st.logout()
        st.rerun()
    st.markdown("---")
    st.caption("Python · Streamlit · SQLite")
    st.caption("TF-IDF · Cosine Similarity")

# -------------------- LOAD RESUME / COMPUTE RESULTS --------------------
saved_resume_bytes = st.session_state.get("saved_resume_bytes")
uploaded_file = io.BytesIO(saved_resume_bytes) if saved_resume_bytes else None
resume_text = ""
resume_error = ""
if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        resume_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not resume_text:
            resume_error = "No readable text found. If this is a scanned PDF, OCR may be required."
    except Exception:
        resume_error = "Could not read this PDF. Please try another text-based PDF."

jobs = pd.DataFrame()
jobs_error = ""
if not JOBS_PATH.exists():
    jobs_error = "jobs.csv is missing. Keep jobs.csv in the same folder as this Python file."
else:
    try:
        jobs = pd.read_csv(JOBS_PATH)
        if not {"Job Role", "Required Skills"}.issubset(jobs.columns):
            jobs_error = 'jobs.csv must contain columns named "Job Role" and "Required Skills".'
            jobs = pd.DataFrame()
        else:
            jobs = jobs.dropna(subset=["Job Role", "Required Skills"]).copy()
    except Exception:
        jobs_error = "Could not read jobs.csv. Check that it is a valid CSV file."

ats_score = keyword_score = section_score = contact_score = 0.0
sections_found = []
detected_skills = []
resume_skills_lower = []
if resume_text:
    resume_lower = resume_text.lower()
    ats_keywords = ["python", "sql", "java", "project", "projects", "education", "skills", "experience", "email", "phone", "contact"]
    keyword_score = sum(k in resume_lower for k in ats_keywords) / len(ats_keywords) * 100
    section_keywords = ["education", "skills", "project", "experience"]
    sections_found = [s for s in section_keywords if s in resume_lower]
    section_score = len(sections_found) / len(section_keywords) * 100
    contact_score = (
        (1 if "@" in resume_text else 0)
        + (1 if any(len(x) >= 10 and x.isdigit() for x in resume_text.replace("-", " ").replace("+", " ").split()) else 0)
    ) / 2 * 100
    ats_score = keyword_score * .45 + section_score * .40 + contact_score * .15
    skill_aliases = {
        "Python": ["python", "python programming"], "SQL": ["sql", "mysql", "postgresql", "sqlite", "structured query language"],
        "Java": ["java", "java programming"], "Pandas": ["pandas"], "NumPy": ["numpy", "numerical python"],
        "Power BI": ["power bi", "powerbi"], "Machine Learning": ["machine learning", "ml model", "supervised learning", "unsupervised learning"],
        "Excel": ["excel", "spreadsheet", "pivot table", "vlookup", "xlookup"], "HTML": ["html", "html5"],
        "CSS": ["css", "css3", "cascading style sheets"], "JavaScript": ["javascript", "js"],
        "OOP": ["oop", "object oriented programming", "object-oriented programming"], "Git": ["git", "github", "gitlab"],
        "Scikit-learn": ["scikit-learn", "sklearn"], "Testing": ["testing", "test cases", "software testing", "manual testing"],
        "Selenium": ["selenium"], "Linux": ["linux", "ubuntu"], "AWS": ["aws", "amazon web services", "ec2", "s3"],
        "Docker": ["docker", "containerization"], "Statistics": ["statistics", "statistical", "probability", "standard deviation"],
        "C++": ["c++", "cpp"], "C": [" c ", " c language"], "FastAPI": ["fastapi"], "Flask": ["flask"],
        "Django": ["django"], "PowerPoint": ["powerpoint", "ppt"], "Communication": ["communication", "presentation skills"]
    }
    detected_skills = [skill for skill, aliases in skill_aliases.items() if any(alias in f" {resume_lower} " for alias in aliases)]
    resume_skills_lower = [s.lower() for s in detected_skills]

if resume_text and not jobs.empty:
    # Hybrid role-fit estimate: required-skill coverage is the main signal;
    # TF-IDF text similarity is a smaller supporting signal. Raw TF-IDF alone
    # often produces tiny percentages (e.g. 2-10%) that are easy to misread.
    try:
        job_texts = jobs["Required Skills"].astype(str).tolist()
        vectors = TfidfVectorizer().fit_transform([resume_text] + job_texts)
        text_scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
    except ValueError:
        text_scores = [0.0] * len(jobs)

    skill_coverage_scores = []
    skill_alias_lookup = {
        "sql": ["sql", "mysql", "postgresql", "sqlite", "structured query language"],
        "power bi": ["power bi", "powerbi"], "machine learning": ["machine learning", "ml model", "supervised learning", "unsupervised learning"],
        "excel": ["excel", "spreadsheet", "pivot table", "vlookup", "xlookup"], "oop": ["oop", "object oriented programming", "object-oriented programming"],
        "scikit-learn": ["scikit-learn", "sklearn"], "aws": ["aws", "amazon web services", "ec2", "s3"],
        "javascript": ["javascript", "js"], "numpy": ["numpy", "numerical python"], "python": ["python", "python programming"]
    }
    def skill_is_present(skill):
        normalized = skill.strip().lower()
        aliases = skill_alias_lookup.get(normalized, [normalized])
        return any(alias in f" {resume_lower} " for alias in aliases)
    for required_text in jobs["Required Skills"].astype(str).tolist():
        required = [x.strip() for x in required_text.split(",") if x.strip()]
        if required:
            matched_count = sum(1 for skill in required if skill_is_present(skill))
            skill_coverage_scores.append(matched_count / len(required))
        else:
            skill_coverage_scores.append(0.0)

    jobs["Skill Coverage"] = [score * 100 for score in skill_coverage_scores]
    jobs["Match Score"] = [
        (coverage * 0.90 + float(text_score) * 100 * 0.10)
        for coverage, text_score in zip(skill_coverage_scores, text_scores)
    ]
    jobs = jobs.sort_values(["Match Score", "Skill Coverage"], ascending=False).reset_index(drop=True)

# -------------------- SHARED PAGE HELPERS --------------------
def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_skill_pills(skills, green=False):
    if not skills:
        st.caption("No skills to show yet.")
        return
    cls = "pill pill-green" if green else "pill"
    st.markdown("".join(f'<span class="{cls}">{s}</span>' for s in skills), unsafe_allow_html=True)


def skill_is_present_for_gap(skill):
    text = resume_text.lower()
    aliases = {"sql": ["sql", "mysql", "postgresql", "sqlite"], "power bi": ["power bi", "powerbi"], "machine learning": ["machine learning", "ml model"], "scikit-learn": ["scikit-learn", "sklearn"], "aws": ["aws", "amazon web services"], "oop": ["oop", "object oriented programming"], "javascript": ["javascript", "js"]}
    return any(a in f" {text} " for a in aliases.get(skill.strip().lower(), [skill.strip().lower()]))

def get_missing_skills():
    if jobs.empty or "Match Score" not in jobs.columns or jobs.empty:
        return [], ""
    top = jobs.iloc[0]
    required = [x.strip() for x in str(top["Required Skills"]).split(",") if x.strip()]
    missing = [x for x in required if not skill_is_present_for_gap(x)]
    return missing, str(top["Job Role"])


# Persist each distinct analysis once per signed-in user, so the History page survives reruns.
if resume_text and not resume_error:
    import hashlib as _history_hashlib
    current_email = st.session_state.logged_in_user["email"].strip().lower()
    resume_hash = _history_hashlib.sha256((st.session_state.get("saved_resume_name", "resume.pdf") + "\n" + resume_text).encode("utf-8")).hexdigest()
    best_role_value = str(jobs.iloc[0]["Job Role"]) if not jobs.empty and "Match Score" in jobs.columns else ""
    best_match_value = float(jobs.iloc[0]["Match Score"]) if not jobs.empty and "Match Score" in jobs.columns else 0.0
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO resume_history (user_email, resume_name, resume_hash, ats_score, skills_count, best_role, match_score) VALUES (?, ?, ?, ?, ?, ?, ?)", (current_email, st.session_state.get("saved_resume_name", "resume.pdf"), resume_hash, float(ats_score), len(detected_skills), best_role_value, best_match_value))
            conn.commit()
    except sqlite3.Error:
        pass

def build_resume_rewrite(text, name, email, target_role):
    """Create a conservative ATS-friendly draft without inventing credentials."""
    clean_lines = [line.strip(" •-\t") for line in text.splitlines() if line.strip()]
    contact = next((line for line in clean_lines if "@" in line), email or "[Email address]")
    heading = (name or (clean_lines[0] if clean_lines else "Your Name")).strip()
    # Keep original facts intact; improve structure and wording only.
    body = "\n".join(clean_lines[:100])
    return f"{heading}\n{contact}\n\nTARGET ROLE\n{target_role or '[Target role]'}\n\nPROFESSIONAL SUMMARY\nMotivated candidate seeking an opportunity to apply the skills and project experience listed below. Customize this summary with your real strengths and measurable achievements before applying.\n\nRESUME CONTENT (organized from your uploaded resume)\n{body}\n\nIMPORTANT: Verify every detail, add measurable results where true, and remove any placeholder text before applying."

def text_to_docx_bytes(text):
    if Document is None:
        return None
    doc = Document()
    for idx, line in enumerate(text.splitlines()):
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.upper() in {"PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "EDUCATION", "PROJECTS", "INTERNSHIP / EXPERIENCE", "CERTIFICATIONS", "RESUME CONTENT (ORGANIZED FROM YOUR UPLOADED RESUME)", "TARGET ROLE"}:
            doc.add_heading(cleaned.title(), level=1)
        elif idx == 0:
            doc.add_heading(cleaned, level=0)
        else:
            doc.add_paragraph(cleaned)
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()

def resume_improvement_suggestions(text, score, detected, sections):
    suggestions = []
    if score < 70: suggestions.append("ATS-style completeness is low: add clearly labelled Skills, Education, Projects, and Experience/Internship sections where applicable.")
    if "@" not in text: suggestions.append("Add a professional email address in the contact section.")
    if not any(ch.isdigit() for ch in text): suggestions.append("Check that your phone number is present and readable.")
    if "project" not in text.lower(): suggestions.append("Add 1–3 relevant projects with technologies used and what you personally implemented.")
    if "education" not in sections: suggestions.append("Add an Education heading with degree, college, graduation year, and CGPA/percentage if appropriate.")
    if "skills" not in sections: suggestions.append("Add a concise Skills section with only skills you can explain in an interview.")
    if "experience" not in sections: suggestions.append("If you have an internship or training, add it under Internship/Experience; do not present training as full-time work.")
    if len(text.split()) < 120: suggestions.append("The extracted resume is short. Consider adding truthful project details, coursework, internship work, and achievements.")
    suggestions.append("Use simple one-column formatting, standard headings, consistent dates, and bullet points. Avoid tables, icons, and skill bars in the final ATS resume.")
    return suggestions

# -------------------- DASHBOARD --------------------
if page == "Dashboard":
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">CAREER WORKSPACE</div>
      <h1>Good to see you, {st.session_state.logged_in_user['name'].split()[0]} 👋</h1>
      <p>Your resume insights and next career steps, all in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    if jobs_error:
        st.error(jobs_error)

    if not uploaded_file:
        st.markdown('<div class="page-title">Start with your resume</div><div class="page-subtitle">Upload a PDF in Resume Analyzer to unlock your personalized dashboard.</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><h3>📄 Resume insights</h3><p>Estimate resume completeness and detect listed skills.</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><h3>🎯 Job matching</h3><p>Compare your resume text with roles in the project dataset.</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><h3>🧩 Skill gap plan</h3><p>Find skills to learn for your best-matching role.</p></div>', unsafe_allow_html=True)
        if st.button("Go to Resume Analyzer →", type="primary"):
            st.session_state.pending_navigation = "Resume Analyzer"
            st.rerun()
    elif resume_error:
        st.error(resume_error)
    else:
        missing_skills, best_role = get_missing_skills()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATS-style score", f"{ats_score:.0f}/100")
        c2.metric("Skills detected", len(detected_skills))
        c3.metric("Roles compared", len(jobs) if not jobs.empty else 0)
        c4.metric("Skills to improve", len(missing_skills))
        st.markdown("### Your resume snapshot")
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.markdown('<div class="card"><h3>✨ Detected skills</h3><p>Skills found by the current keyword-based detector.</p></div>', unsafe_allow_html=True)
            render_skill_pills(detected_skills, green=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card"><h3>🧭 Recommended next step</h3><p>Use the Skill Gap Plan to turn missing skills into a learning checklist.</p></div>', unsafe_allow_html=True)
            if st.button("View my skill gap →", type="primary"):
                st.session_state.pending_navigation = "Skill Gap Plan"
                st.rerun()
        with right:
            st.markdown('<div class="card"><h3>🎯 Best matching role</h3><p>Based on required-skill coverage and resume text similarity.</p></div>', unsafe_allow_html=True)
            if not jobs.empty:
                best = jobs.iloc[0]
                st.metric(str(best["Job Role"]), f"{best['Match Score']:.0f}% skill-based fit")
                st.progress(max(0.0, min(1.0, float(best["Match Score"]) / 100)))
                st.caption("Score uses 90% required-skill coverage and 10% resume-text similarity. It is an educational estimate, not a hiring probability.")
            if st.button("Explore all job matches →"):
                st.session_state.pending_navigation = "Job Matches"
                st.rerun()

# -------------------- RESUME ANALYZER --------------------
elif page == "Resume Analyzer":
    page_header("Resume Analyzer", "Upload a text-based PDF to generate your resume insights.")
    st.markdown('<div class="card"><h3>📤 Upload your resume</h3><p>PDF only · Use a readable, text-based PDF for best results.</p></div>', unsafe_allow_html=True)
    st.file_uploader("Choose PDF resume", type=["pdf"], key="resume_uploader_widget", on_change=persist_resume_upload)
    saved_resume_bytes = st.session_state.get("saved_resume_bytes")
    uploaded_file = io.BytesIO(saved_resume_bytes) if saved_resume_bytes else None
    if uploaded_file is None:
        st.info("Choose a PDF above. Your analysis will appear on this page and in the dashboard.")
    elif resume_error:
        st.error(resume_error)
    elif resume_text:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATS-style score", f"{ats_score:.0f}/100")
        c2.metric("Keyword coverage", f"{keyword_score:.0f}%")
        c3.metric("Section coverage", f"{section_score:.0f}%")
        c4.metric("Contact check", f"{contact_score:.0f}%")
        st.progress(int(ats_score) / 100)
        if ats_score >= 80:
            st.success("Good basic coverage. Review the job-specific match as well.")
        elif ats_score >= 60:
            st.warning("Moderate coverage. Check the missing sections and keywords.")
        else:
            st.warning("Several basic checks may be missing. Review the resume sections and contact details.")
        st.markdown("### Skills detected")
        render_skill_pills(detected_skills, green=True)
        with st.expander("How is this ATS-style score calculated?"):
            st.write("Educational estimate: keyword coverage (45%), common resume section coverage (40%), and basic contact checks (15%). It is not an official score from a commercial ATS.")
            st.write(f"Keywords found: {sum(k in resume_text.lower() for k in ['python','sql','java','project','projects','education','skills','experience','email','phone','contact'])} of 11.")
            st.write("Sections found: " + (", ".join(sections_found) if sections_found else "None detected"))
        with st.expander("Preview extracted resume text"):
            st.text(resume_text[:12000] + ("\n\n[Preview truncated]" if len(resume_text) > 12000 else ""))

# -------------------- JOB MATCHES --------------------
elif page == "Job Matches":
    page_header("Job Matches", "Explore the top roles ranked against your resume.")
    if not uploaded_file:
        st.info("Upload your resume in Resume Analyzer first to see personalized job matches.")
        if st.button("Open Resume Analyzer →", type="primary"):
            st.session_state.pending_navigation = "Resume Analyzer"
            st.rerun()
    elif resume_error:
        st.error(resume_error)
    elif jobs_error or jobs.empty:
        st.error(jobs_error or "No jobs found in jobs.csv.")
    else:
        st.caption("Score emphasizes required-skill coverage (90%) and uses text similarity (10%) as a secondary signal. The percentage is an estimate, not a hiring probability.")
        for rank, (_, row) in enumerate(jobs.head(3).iterrows(), start=1):
            required = [x.strip() for x in str(row["Required Skills"]).split(",") if x.strip()]
            matched = [x for x in required if x.lower() in resume_skills_lower]
            missing = [x for x in required if not skill_is_present_for_gap(x)]
            with st.container(border=True):
                lcol, rcol = st.columns([3, 1])
                with lcol:
                    st.markdown(f"### {rank:02d} · {row['Job Role']}")
                    st.caption(f"{len(matched)}/{len(required)} required skills found · {len(matched) / len(required) * 100:.0f}% skill coverage" if required else "No required skills listed for this role")
                with rcol:
                    st.metric("Role fit", f"{row['Match Score']:.0f}%")
                st.progress(max(0.0, min(1.0, float(row["Match Score"]) / 100)))
                mcol, gcol = st.columns(2)
                with mcol:
                    st.markdown("**Matched skills**")
                    render_skill_pills(matched, green=True)
                with gcol:
                    st.markdown("**Skills to build**")
                    render_skill_pills(missing)
                with st.expander("Why is this role recommended?"):
                    st.write(f"This estimate combines required-skill coverage (90%) and resume text similarity (10%). Your resume contains {len(matched)} of {len(required)} listed skills. This is an educational estimate, not a probability of getting hired.")

# -------------------- SKILL GAP PLAN --------------------
elif page == "Skill Gap Plan":
    page_header("Skill Gap Plan", "Turn missing skills into a practical learning checklist.")
    if not uploaded_file:
        st.info("Upload your resume in Resume Analyzer first.")
        if st.button("Open Resume Analyzer →", type="primary"):
            st.session_state.pending_navigation = "Resume Analyzer"
            st.rerun()
    elif resume_error:
        st.error(resume_error)
    elif jobs_error or jobs.empty:
        st.error(jobs_error or "No jobs found in jobs.csv.")
    else:
        missing_skills, best_role = get_missing_skills()
        st.markdown(f'<div class="card"><h3>🎯 Best matching role: {best_role}</h3><p>These are skills from that role that were not detected in the uploaded resume.</p></div>', unsafe_allow_html=True)
        if not missing_skills:
            st.success("All listed skills for the best-matching role were detected. Tailor your project descriptions to the job as a next step.")
        else:
            render_skill_pills(missing_skills)
            improvement_plan = {
                "Python": ("High", "Build a small practical Python project and practise functions, lists, dictionaries, and file handling.", "Describe what you built and your contribution in 2–3 resume bullets."),
                "SQL": ("High", "Practise JOINs, GROUP BY, subqueries, and aggregate functions.", "Mention a database task or query-based project you completed."),
                "Power BI": ("High", "Create an interactive dashboard using a public dataset.", "Explain the dashboard insights and visuals in your project section."),
                "Excel": ("Medium", "Practise Pivot Tables, lookup functions, formulas, and charts.", "Mention practical Excel skills you can demonstrate."),
                "Pandas": ("Medium", "Practise reading CSV files, filtering data, and handling missing values.", "Describe a data-cleaning task you completed."),
                "NumPy": ("Low", "Practise arrays, indexing, and mathematical operations.", "Mention NumPy after using it in a practical task."),
                "Machine Learning": ("High", "Build a small prediction project with Scikit-learn.", "Name the dataset, algorithm, and evaluation metric."),
                "Statistics": ("Medium", "Review mean, median, mode, standard deviation, correlation, and probability.", "Explain where you applied statistics."),
                "Git": ("Medium", "Practise add, commit, push, pull, and branches.", "Link your completed project repository if available."),
                "AWS": ("Medium", "Learn the basics of EC2, S3, and IAM.", "Mention only AWS work you have actually completed."),
                "Docker": ("Low", "Learn images, containers, Dockerfiles, and basic commands.", "Add it to your resume after completing a practical task."),
                "Java": ("Medium", "Practise OOP, collections, exception handling, and problem-solving.", "Describe a Java project if completed."),
                "HTML": ("Low", "Practise forms, tables, links, and page structure.", "Mention where you used HTML in a real project."),
                "CSS": ("Low", "Practise layouts, spacing, and responsive design.", "Show how you used CSS in a project."),
                "JavaScript": ("Medium", "Learn variables, functions, arrays, objects, and DOM basics.", "Add a project when you have practical experience."),
                "OOP": ("Medium", "Review classes, objects, inheritance, encapsulation, and polymorphism.", "Describe where you applied OOP."),
                "Scikit-learn": ("Medium", "Practise train/test split, model fitting, and evaluation.", "Mention the estimator and metric used in your project."),
                "Testing": ("Medium", "Learn test cases, expected versus actual results, and regression testing.", "Describe test cases you created."),
                "Selenium": ("Low", "Automate a simple browser workflow and learn locators.", "Describe the workflow you automated."),
                "Linux": ("Medium", "Practise navigation, file commands, permissions, and processes.", "Mention tasks you have actually performed."),
            }
            st.markdown("### Recommended actions")
            for skill in missing_skills:
                priority, action, resume_tip = improvement_plan.get(
                    skill, ("Medium", f"Learn {skill} and complete a small practical exercise.", f"Add {skill} only after gaining practical experience.")
                )
                with st.container(border=True):
                    pcol, dcol = st.columns([1, 4])
                    with pcol:
                        st.markdown(f"**{priority} priority**")
                        st.markdown(f"### {skill}")
                    with dcol:
                        st.write(action)
                        st.caption("Resume tip: " + resume_tip)

# -------------------- RESUME IMPROVEMENT --------------------
elif page == "Resume Improvement":
    page_header("Resume Improvement", "Find issues, get practical fixes, and download a cleaner draft.")
    if not resume_text:
        st.info("Upload your PDF in Resume Analyzer first. Your suggestions will appear here.")
        if st.button("Open Resume Analyzer →", type="primary", key="improve_open_analyzer"):
            st.session_state.active_page = "Resume Analyzer"
            st.rerun()
    elif resume_error:
        st.error(resume_error)
    else:
        st.metric("Current ATS-style completeness", f"{ats_score:.0f}/100")
        suggestions = resume_improvement_suggestions(resume_text, ats_score, detected_skills, sections_found)
        st.markdown("### Recommended improvements")
        for idx, suggestion in enumerate(suggestions, start=1):
            st.markdown(f"**{idx}.** {suggestion}")
        st.markdown("### ATS-friendly rewrite draft")
        user = st.session_state.logged_in_user
        target_role = st.text_input("Target role for the rewritten resume", value=(str(jobs.iloc[0]["Job Role"]) if not jobs.empty and "Match Score" in jobs.columns else "Entry-level role"), key="rewrite_target_role")
        rewritten = build_resume_rewrite(resume_text, user.get("name", ""), user.get("email", ""), target_role)
        st.text_area("Review and edit before using", rewritten, height=360, key="rewritten_resume_text")
        final_rewrite = st.session_state.get("rewritten_resume_text", rewritten)
        st.download_button("⬇ Download rewritten resume draft (.txt)", data=final_rewrite, file_name="ResumeIQ_Resume_Draft.txt", mime="text/plain", use_container_width=True)
        docx_data = text_to_docx_bytes(final_rewrite)
        if docx_data:
            st.download_button("⬇ Download editable Word resume (.docx)", data=docx_data, file_name="ResumeIQ_Resume_Draft.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        st.caption("This is a structured draft based on extracted text, not a guarantee of ATS success. Verify all facts and replace generic summary wording before applying.")

# -------------------- RESUME BUILDER --------------------
elif page == "Resume Builder":
    page_header("Resume Builder", "Create a clean, editable resume draft using your real information.")
    st.info("Fill only details that are true. The app will not invent work experience, skills, or achievements.")
    with st.form("resume_builder_form"):
        b_name = st.text_input("Full name", value=st.session_state.logged_in_user.get("name", ""))
        b_email = st.text_input("Email", value=st.session_state.logged_in_user.get("email", ""))
        b_phone = st.text_input("Phone number")
        b_location = st.text_input("City, state")
        b_role = st.text_input("Target role", value="Entry-Level Analyst / Developer")
        b_skills = st.text_area("Skills (comma separated)", placeholder="Python, SQL, Excel")
        b_education = st.text_area("Education", placeholder="Degree, college, year, CGPA/percentage")
        b_projects = st.text_area("Projects", placeholder="Project name, technologies, your contribution, result")
        b_internship = st.text_area("Internship / experience (if any)")
        b_certifications = st.text_area("Certifications (if any)")
        build_clicked = st.form_submit_button("Generate resume draft", type="primary")
    if build_clicked:
        sections = [b_name, f"{b_email} | {b_phone} | {b_location}", "", "PROFESSIONAL SUMMARY", f"Entry-level candidate targeting {b_role}. Add 1–2 truthful sentences describing your strongest relevant skills and goals.", "", "TECHNICAL SKILLS", b_skills or "[Add only skills you can demonstrate]", "", "EDUCATION", b_education or "[Add degree, institution, and graduation details]", "", "PROJECTS", b_projects or "[Add project details and your contribution]", "", "INTERNSHIP / EXPERIENCE", b_internship or "[Add relevant internship/training, or remove this section if not applicable]", "", "CERTIFICATIONS", b_certifications or "[Add relevant certifications, or remove this section]", ""]
        builder_text = "\n".join(sections)
        st.session_state.builder_output = builder_text
    if st.session_state.get("builder_output"):
        st.text_area("Generated resume — review before applying", st.session_state.builder_output, height=400, key="builder_output_editor")
        final_builder = st.session_state.get("builder_output_editor", st.session_state.builder_output)
        st.download_button("⬇ Download resume draft (.txt)", data=final_builder, file_name="ResumeIQ_Built_Resume.txt", mime="text/plain")
        builder_docx = text_to_docx_bytes(final_builder)
        if builder_docx:
            st.download_button("⬇ Download editable Word resume (.docx)", data=builder_docx, file_name="ResumeIQ_Built_Resume.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        st.caption("For college demo, this provides a downloadable editable draft. Remove placeholders and check every statement before submitting it to employers.")

# -------------------- RESUME HISTORY --------------------
elif page == "Resume History":
    page_header("Resume History", "See previous resume analyses saved for your account.")
    user_email = st.session_state.logged_in_user["email"].strip().lower()
    try:
        with get_db_connection() as conn:
            history = pd.read_sql_query("SELECT resume_name AS 'Resume file', ats_score AS 'ATS-style score', skills_count AS 'Skills detected', best_role AS 'Best matching role', match_score AS 'Estimated fit %', created_at AS 'Analyzed at' FROM resume_history WHERE user_email = ? ORDER BY created_at DESC", conn, params=(user_email,))
        if history.empty:
            st.info("No saved analyses yet. Upload a resume in Resume Analyzer and it will appear here.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
            csv_bytes = history.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download analysis history (.csv)", data=csv_bytes, file_name="ResumeIQ_Analysis_History.csv", mime="text/csv")
    except Exception:
        st.warning("Could not load history. Please restart the app and try again.")

# -------------------- ACCOUNT --------------------
elif page == "My Account":
    page_header("My Account", "Your local project account details.")
    user = st.session_state.logged_in_user
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><h3>Profile</h3><p>Account details currently stored in the local SQLite database.</p></div>', unsafe_allow_html=True)
        st.text_input("Full name", value=user["name"], disabled=True)
        st.text_input("Email address", value=user["email"], disabled=True)
    with c2:
        st.markdown('<div class="card"><h3>Privacy & project notes</h3><p>Your account database is stored beside app.py on this computer.</p></div>', unsafe_allow_html=True)
        st.info("Passwords are stored as salted hashes, not plain text. This is a local educational prototype, not a production identity system.")
        st.caption("Analysis summaries are saved locally in resume_users.db. Uploaded PDF bytes stay in the current app session; the app stores analysis metadata, not the original PDF file.")
    st.markdown("### Recent resume analysis history")
    try:
        with get_db_connection() as conn:
            account_history = pd.read_sql_query("SELECT resume_name AS 'Resume file', ats_score AS 'ATS score', best_role AS 'Best role', created_at AS 'Date' FROM resume_history WHERE user_email = ? ORDER BY created_at DESC LIMIT 10", conn, params=(user["email"].strip().lower(),))
        if account_history.empty:
            st.info("Your resume analysis history will appear here after you upload a PDF.")
        else:
            st.dataframe(account_history, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download profile history (.csv)", data=account_history.to_csv(index=False).encode("utf-8"), file_name="ResumeIQ_Profile_History.csv", mime="text/csv")
    except Exception:
        st.caption("History is not available right now.")

st.markdown('<div class="footer">ResumeIQ · AI-Powered Resume Screening & Job Recommendation System<br>Educational prototype · Scores are estimates, not hiring guarantees.</div>', unsafe_allow_html=True)
