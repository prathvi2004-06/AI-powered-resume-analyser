import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import hmac
import secrets
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(180deg,#f6f8fc,#edf3fb);}
.block-container {max-width:1180px;padding-top:1.5rem;padding-bottom:3rem;}
.hero {padding:1.7rem 2rem;border-radius:22px;background:linear-gradient(120deg,#142b50,#2466a6,#3b8bc8);color:white;margin-bottom:1.2rem;}
.hero h1 {color:white;margin:0;font-size:2.05rem;}
.hero p {color:#e5efff;margin:.5rem 0 0;}
.section-title {color:#183153;font-size:1.35rem;font-weight:750;margin-top:1rem;margin-bottom:.35rem;}
.small-muted {color:#64748b;font-size:.9rem;}
div[data-testid="stMetric"] {background:white;border:1px solid #e1e8f2;padding:14px 16px;border-radius:15px;box-shadow:0 3px 12px #1e375a0d;}
div[data-testid="stFileUploader"] {background:white;padding:1rem;border:1px dashed #8aa8cf;border-radius:15px;}
.skill-chip {display:inline-block;background:#e7f5ee;color:#176343;border:1px solid #bce5d0;border-radius:20px;padding:5px 10px;margin:4px;font-size:.88rem;}
.footer-note {color:#64748b;font-size:.82rem;text-align:center;padding-top:1.3rem;}
</style>
""", unsafe_allow_html=True)



# -------------------- USER LOGIN / SIGN-UP DATABASE --------------------
# SQLite creates resume_users.db automatically in the same folder as app.py.
DB_PATH = Path(__file__).with_name("resume_users.db")


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
        conn.commit()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()
    return password_hash, salt


def create_user(full_name, email, password):
    password_hash, salt = hash_password(password)
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                (full_name.strip(), email.strip().lower(), password_hash, salt)
            )
            conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except sqlite3.Error:
        return False, "Could not create the account because of a database error."


def verify_user(email, password):
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT id, full_name, email, password_hash, salt FROM users WHERE email = ?",
                (email.strip().lower(),)
            ).fetchone()
        if not row:
            return None
        user_id, full_name, saved_email, saved_hash, salt = row
        entered_hash, _ = hash_password(password, salt)
        if hmac.compare_digest(entered_hash, saved_hash):
            return {"id": user_id, "name": full_name, "email": saved_email}
        return None
    except sqlite3.Error:
        return None