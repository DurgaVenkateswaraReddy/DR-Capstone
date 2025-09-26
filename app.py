# app.py
# Flask + MongoDB (Compass) + Ensemble inference (exactly like your training script)
import os
import io
import json
import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, abort
)
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError
from bson.objectid import ObjectId

import pandas as pd
import numpy as np
from joblib import load
from xhtml2pdf import pisa

# --------------- Flask config ---------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# Copy your connection string from MongoDB Compass (Driver: Python, Version: 3.12+)
# Example Atlas (with db in URI): mongodb+srv://user:pass@cluster.mongodb.net/dr_app?retryWrites=true&w=majority
# Example local (no db in URI): mongodb://localhost:27017
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "dr_app")

bcrypt = Bcrypt(app)

# --------------- MongoDB (Compass-friendly) ---------------
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")  # test connection
except ServerSelectionTimeoutError as e:
    app.logger.error(f"Could not connect to MongoDB: {e}")
    raise

# If the URI includes a default DB (/mydb), use it; otherwise fall back to MONGO_DB_NAME
try:
    db = client.get_default_database()
    if not db or db.name in (None, "", "admin"):
        raise ConfigurationError("No default DB in URI")
except Exception:
    db = client[MONGO_DB_NAME]

users_col = db["users"]
reports_col = db["reports"]

# Indexes
users_col.create_index("username", unique=True)
reports_col.create_index([("probability", DESCENDING), ("created_at", DESCENDING)])

# --------------- Auth ---------------
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, _id, username, name, role):
        self.id = str(_id)
        self.username = username
        self.name = name
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    try:
        doc = users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    if not doc:
        return None
    return User(doc["_id"], doc["username"], doc["name"], doc["role"])

# --------------- ML artifacts (ensemble) ---------------
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"

SCALER = None
FEATURES = []
CLINICAL_FEATURES = []
CLINICAL_FACTOR = 0.1
ENSEMBLE_THRESHOLD = 0.5
MODEL_ORDER = []
MODELS = {}
ENSEMBLE_LABEL = ""

def load_artifacts():
    global SCALER, FEATURES, CLINICAL_FEATURES, CLINICAL_FACTOR
    global ENSEMBLE_THRESHOLD, MODEL_ORDER, MODELS, ENSEMBLE_LABEL

    meta_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
    if not os.path.exists(meta_path) or not os.path.exists(scaler_path):
        app.logger.warning("Artifacts missing. Run train.py first.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    FEATURES = meta["features"]
    CLINICAL_FEATURES = meta.get("clinical_features", ["fasting_glucose","hba1c","diabetes_duration"])
    CLINICAL_FACTOR = float(meta.get("clinical_scale_factor", 0.1))
    ENSEMBLE_THRESHOLD = float(meta["ensemble_threshold"])
    MODEL_ORDER = meta["model_order"]
    model_filenames = meta["model_filenames"]

    SCALER = load(scaler_path)
    MODELS = {}
    for name in MODEL_ORDER:
        fname = model_filenames[name]
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            app.logger.error(f"Missing model file: {path}")
            continue
        MODELS[name] = load(path)

    ENSEMBLE_LABEL = f"Ensemble(mean) of {len(MODELS)} models"

def compute_ensemble_probability(feature_dict: dict) -> float:
    """
    EXACT inference steps matching your training code:
    - Features in same order
    - Multiply clinical features by 0.1
    - Transform with saved StandardScaler (fit before split on resampled X)
    - Predict proba for each model and average
    """
    if SCALER is None or not MODELS:
        raise RuntimeError("Model artifacts not loaded. Run train.py first.")

    X = pd.DataFrame([feature_dict], columns=FEATURES).copy()
    X.loc[:, CLINICAL_FEATURES] = X.loc[:, CLINICAL_FEATURES].astype(float) * CLINICAL_FACTOR
    X_scaled = SCALER.transform(X)

    probs = []
    for name in MODEL_ORDER:
        mdl = MODELS.get(name)
        if mdl is None:
            continue
        p = float(mdl.predict_proba(X_scaled)[:, 1][0])
        probs.append(p)
    if not probs:
        raise RuntimeError("No models loaded.")
    return float(np.mean(probs))

def html_to_pdf(html_content: str, filename: str):
    pdf_io = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html_content), dest=pdf_io)
    if pisa_status.err:
        return None
    pdf_io.seek(0)
    return send_file(pdf_io, as_attachment=True, download_name=filename, mimetype="application/pdf")

# Load ML artifacts on startup
load_artifacts()

# --------------- Routes ---------------
@app.route("/")
def index():
    return render_template("index.html", model_name=ENSEMBLE_LABEL)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        role = request.form.get("role")
        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if role not in ("patient", "doctor"):
            flash("Please select a valid role.", "danger")
            return redirect(url_for("register"))
        if not name or not username or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if users_col.find_one({"username": username}):
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        doc = {
            "role": role, "name": name, "username": username,
            "password_hash": pw_hash, "created_at": datetime.datetime.utcnow(),
        }
        res = users_col.insert_one(doc)
        user = User(res.inserted_id, username, name, role)
        login_user(user)
        flash("Registered and logged in.", "success")
        return redirect(url_for("patient_dashboard" if role == "patient" else "doctor_dashboard"))
    return render_template("register.html", model_name=ENSEMBLE_LABEL)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        doc = users_col.find_one({"username": username})
        if not doc or not bcrypt.check_password_hash(doc["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        user = User(doc["_id"], doc["username"], doc["name"], doc["role"])
        login_user(user)
        flash("Logged in.", "success")
        return redirect(url_for("patient_dashboard" if doc["role"] == "patient" else "doctor_dashboard"))
    return render_template("login.html", model_name=ENSEMBLE_LABEL)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

@app.route("/patient")
@login_required
def patient_dashboard():
    if current_user.role != "patient":
        abort(403)
    pid = ObjectId(current_user.id)
    reports = list(reports_col.find({"patient_id": pid}).sort("created_at", DESCENDING))
    for r in reports:
        did = r.get("doctor_id")
        if did:
            ddoc = users_col.find_one({"_id": did}) or {}
            r["doctor_name"] = ddoc.get("name", "Unknown")
        else:
            r["doctor_name"] = "Not assigned"
        r["id_str"] = str(r["_id"])
    return render_template("patient_dashboard.html", user=current_user, reports=reports, model_name=ENSEMBLE_LABEL)

@app.route("/assessment/new", methods=["GET", "POST"])
@login_required
def new_assessment():
    if current_user.role != "patient":
        abort(403)
    doctors = list(users_col.find({"role": "doctor"}).sort("name", ASCENDING))
    if request.method == "POST":
        try:
            feature_values = {
                "exudates_count": int(request.form["exudates_count"]),
                "hemorrhages_count": int(request.form["hemorrhages_count"]),
                "microaneurysms_count": int(request.form["microaneurysms_count"]),
                "vessel_tortuosity": float(request.form["vessel_tortuosity"]),
                "macular_thickness": float(request.form["macular_thickness"]),
                "fasting_glucose": float(request.form["fasting_glucose"]),
                "hba1c": float(request.form["hba1c"]),
                "diabetes_duration": float(request.form["diabetes_duration"]),
            }
        except Exception:
            flash("Please enter valid numeric values.", "danger")
            return redirect(url_for("new_assessment"))

        doctor_id_str = request.form.get("doctor_id") or ""
        doctor_id = ObjectId(doctor_id_str) if doctor_id_str else None

        try:
            prob = compute_ensemble_probability(feature_values)
        except Exception as e:
            app.logger.exception(e)
            flash("Model artifacts missing. Train the model first.", "danger")
            return redirect(url_for("patient_dashboard"))

        pred = int(prob >= ENSEMBLE_THRESHOLD)
        report_doc = {
            "patient_id": ObjectId(current_user.id),
            "doctor_id": doctor_id,
            "features": feature_values,
            "probability": prob,
            "prediction": pred,
            "threshold": ENSEMBLE_THRESHOLD,
            "model_name": ENSEMBLE_LABEL,
            "created_at": datetime.datetime.utcnow(),
        }
        reports_col.insert_one(report_doc)
        flash("Assessment created.", "success")
        return redirect(url_for("patient_dashboard"))
    return render_template("report_form.html", doctors=doctors, model_name=ENSEMBLE_LABEL)

@app.route("/doctor")
@login_required
def doctor_dashboard():
    if current_user.role != "doctor":
        abort(403)
    order = request.args.get("order", "desc")
    sort_dir = DESCENDING if order == "desc" else ASCENDING
    reports = list(reports_col.find({}).sort("probability", sort_dir))
    cache = {}
    for r in reports:
        pid = r["patient_id"]
        did = r.get("doctor_id")
        if pid not in cache:
            pdoc = users_col.find_one({"_id": pid}) or {}
            cache[pid] = pdoc.get("name", "Unknown")
        if did and did not in cache:
            ddoc = users_col.find_one({"_id": did}) or {}
            cache[did] = ddoc.get("name", "Unknown")
        r["patient_name"] = cache.get(pid, "Unknown")
        r["doctor_name"] = cache.get(did, "Not assigned")
        r["id_str"] = str(r["_id"])
    return render_template("doctor_dashboard.html", reports=reports, order=order, model_name=ENSEMBLE_LABEL)

@app.route("/report/<rid>/pdf")
@login_required
def report_pdf(rid):
    try:
        report = reports_col.find_one({"_id": ObjectId(rid)})
    except Exception:
        abort(404)
    if not report:
        abort(404)

    if current_user.role == "patient" and report["patient_id"] != ObjectId(current_user.id):
        abort(403)

    patient = users_col.find_one({"_id": report["patient_id"]}) or {}
    doctor = users_col.find_one({"_id": report.get("doctor_id")}) if report.get("doctor_id") else {}
    context = {
        "patient_name": patient.get("name", "Unknown"),
        "patient_username": patient.get("username", ""),
        "doctor_name": (doctor or {}).get("name", "Not assigned"),
        "doctor_username": (doctor or {}).get("username", ""),
        "created_at": report["created_at"].strftime("%Y-%m-%d %H:%M"),
        "features": report["features"],
        "probability": report["probability"],
        "prediction": report["prediction"],
        "threshold": report["threshold"],
        "model_name": report["model_name"],
    }
    html = render_template("report_pdf.html", **context)
    pdf_io = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf_io)
    if pisa_status.err:
        flash("Failed to generate PDF.", "danger")
        return redirect(url_for("patient_dashboard" if current_user.role == "patient" else "doctor_dashboard"))
    pdf_io.seek(0)
    return send_file(pdf_io, as_attachment=True, download_name=f"DR_Report_{rid}.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)