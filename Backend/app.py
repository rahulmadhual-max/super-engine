"""
AI Resume Analyzer — Flask Backend
====================================
Production-ready REST API with JWT auth and MongoDB Atlas.
"""

import os
import io
from datetime import timedelta

import pdfplumber
from analyzer import analyze_resume
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity,
)
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

load_dotenv()

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = Flask(__name__)

#  FIXED CORS (FULL)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/resume_analyzer")

# ─────────────────────────────────────────────
# EXTENSIONS
# ─────────────────────────────────────────────
bcrypt = Bcrypt(app)
jwt    = JWTManager(app)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True
)

db = client["resume_analyzer"]
users = db["users"]
users.create_index("email", unique=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def err(msg, code=400):
    return jsonify({"success": False, "error": msg}), code

def ok(data, code=200):
    return jsonify({"success": True, **data}), code

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return ok({"message": "AI Resume Analyzer API is running."})

# ── REGISTER ─────────────────────────────────
@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return ok({})

    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return err("Request body must be JSON or form data.")

    name     = (data.get("name")     or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return err("Email and password are required.")
    if len(password) < 6:
        return err("Password must be at least 6 characters.")

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")

    try:
        users.insert_one({
            "name":        name,
            "email":       email,
            "password":    hashed,
            "first_login": True,
        })
    except DuplicateKeyError:
        return err("An account with this email already exists.", 409)

    return ok({
        "message": "Welcome! Your account has been created successfully.",
    }, 201)

# ── LOGIN ─────────────────────────────────────
@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return ok({})

    data = request.get_json(silent=True) or request.form.to_dict()

    if not data:
        return err("Email and password are required.")

    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return err("Email and password are required.")

    user = users.find_one({"email": email})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return err("Invalid email or password.", 401)

    name = user.get("name", "").strip()
    if user.get("first_login", True):
        message = f"Welcome, {name}! 👋" if name else "Welcome! 👋"
        users.update_one({"email": email}, {"$set": {"first_login": False}})
    else:
        message = f"Welcome back, {name}! 👋" if name else "Welcome back! 👋"

    token = create_access_token(identity=email)

    return ok({
        "token":   token,
        "message": message,
        "email":   email,
        "name":    name,
    })

# ── ANALYZE ───────────────────────────────────
@app.route("/analyze", methods=["POST", "OPTIONS"])
@jwt_required()
def analyze():
    if request.method == "OPTIONS":
        return ok({})

    current_user = get_jwt_identity()

    job_description = request.form.get("job_description", "").strip()
    resume_file     = request.files.get("resume")

    if not resume_file or not job_description:
        return err("Both 'resume' and 'job_description' are required.")

    if not resume_file.filename.lower().endswith(".pdf"):
        return err("Only PDF files are accepted.")

    try:
        pdf_bytes   = resume_file.read()
        resume_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text + "\n"
    except Exception as e:
        return err(f"Could not read PDF: {str(e)}")

    if not resume_text.strip():
        return err("No text could be extracted from the PDF.")

    try:
        result = analyze_resume(resume_text, job_description)
    except Exception as e:
        return err(f"Analysis error: {str(e)}")

    return ok({
        "analyzed_by":    current_user,
        "score":          result["score"],
        "missing_skills": result["missing_skills"],
        "strengths":      result["strengths"],
        "suggestions":    result["suggestions"],
        "breakdown":      result["breakdown"],
        "resume_length":  len(resume_text.split()),
    })

# ─────────────────────────────────────────────
# JWT ERRORS
# ─────────────────────────────────────────────
@jwt.unauthorized_loader
def unauthorized_cb(reason):
    return err("Authorization token missing or invalid.", 401)

@jwt.expired_token_loader
def expired_cb(header, data):
    return err("Session expired.", 401)

@jwt.invalid_token_loader
def invalid_cb(reason):
    return err("Invalid token.", 401)

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)