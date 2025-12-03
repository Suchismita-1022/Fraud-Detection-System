# app.py
import os
import sqlite3
import threading
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "app_data.db"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
MANUAL_REVIEW_FLAG = "manual_review"
VERIFIED_FLAG = "verified"
FAILED_FLAG = "failed"
PROCESSING_FLAG = "processing"

# Estimated manual review time per document (seconds); used to compute time saved
ESTIMATED_MANUAL_REVIEW_SEC = 5 * 60  # 5 minutes

# Ensure directories exist
for directory in (UPLOADS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utilities: DB and helpers
# ----------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Users table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS users
                (
                    id
                    TEXT
                    PRIMARY
                    KEY,
                    username
                    TEXT
                    UNIQUE
                    NOT
                    NULL,
                    password_hash
                    TEXT
                    NOT
                    NULL,
                    created_at
                    TEXT
                    NOT
                    NULL
                )
                """)
    # Submissions table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS submissions
                (
                    id
                    TEXT
                    PRIMARY
                    KEY,
                    user_id
                    TEXT,
                    filename
                    TEXT,
                    filepath
                    TEXT,
                    uploaded_at
                    TEXT,
                    status
                    TEXT,
                    processing_time_sec
                    REAL
                    DEFAULT
                    0,
                    automated_timestamp
                    TEXT,
                    verifier_notes
                    TEXT,
                    time_saved_sec
                    REAL
                    DEFAULT
                    0
                )
                """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), username, hash_password(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate(username: str, password: str) -> Optional[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row["password_hash"] == hash_password(password):
        return dict(row)
    return None


def add_submission(user_id: str, filename: str, filepath: str) -> str:
    submission_id = str(uuid.uuid4())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
                INSERT INTO submissions (id, user_id, filename, filepath, uploaded_at, status, automated_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (submission_id, user_id, filename, filepath, datetime.utcnow().isoformat(), PROCESSING_FLAG, None))
    conn.commit()
    conn.close()
    return submission_id


def update_submission_status(submission_id: str, status: str, processing_time_sec: float = 0.0, notes: str = ""):
    conn = get_db_connection()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    time_saved = 0.0
    if status == VERIFIED_FLAG or status == MANUAL_REVIEW_FLAG:
        # estimate time saved = manual_time - actual_processing_time (clamped >= 0)
        time_saved = max(0.0, ESTIMATED_MANUAL_REVIEW_SEC - processing_time_sec)
        cur.execute("""
                    UPDATE submissions
                    SET status              = ?,
                        processing_time_sec = ?,
                        automated_timestamp = ?,
                        verifier_notes      = ?,
                        time_saved_sec      = ?
                    WHERE id = ?
                    """, (status, processing_time_sec, now, notes, time_saved, submission_id))
    else:
        cur.execute("""
                    UPDATE submissions
                    SET status              = ?,
                        processing_time_sec = ?,
                        automated_timestamp = ?,
                        verifier_notes      = ?,
                        time_saved_sec      = ?
                    WHERE id = ?
                    """, (status, processing_time_sec, now, notes, time_saved, submission_id))
    conn.commit()
    conn.close()


def get_user_submissions(user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM submissions WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_system_stats():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as total FROM submissions")
    total = cur.fetchone()["total"]
    cur.execute("SELECT COUNT(*) as verified FROM submissions WHERE status = ?", (VERIFIED_FLAG,))
    verified = cur.fetchone()["verified"]
    cur.execute("SELECT COUNT(*) as manual FROM submissions WHERE status = ?", (MANUAL_REVIEW_FLAG,))
    manual = cur.fetchone()["manual"]
    cur.execute("SELECT COUNT(*) as processing FROM submissions WHERE status = ?", (PROCESSING_FLAG,))
    processing = cur.fetchone()["processing"]
    cur.execute("SELECT SUM(time_saved_sec) as total_time_saved FROM submissions")
    row = cur.fetchone()
    total_time_saved = row["total_time_saved"] or 0.0
    conn.close()
    return {
        "total": total,
        "verified": verified,
        "manual": manual,
        "processing": processing,
        "total_time_saved_sec": total_time_saved
    }


# ----------------------------
# Simulated verification worker
# Replace or extend this to call your actual backend inference pipeline
# ----------------------------
_worker_lock = threading.Lock()
_worker_started = False


def simulated_verify_worker(submission_id: str, filepath: str):
    """
    Simulated verification: sleeps for short time then decides status.
    Replace this by calling your detection/classification pipeline (YOLO+ResNet) and update DB accordingly.
    """
    start = time.time()
    # Simulate variable processing time
    proc_time = 2 + (uuid.uuid4().int % 5)  # between 2 and 6 seconds
    time.sleep(proc_time)

    # Heuristic: if filename contains 'forged' (case-insensitively) mark manual_review; else verified
    filename = Path(filepath).name.lower()
    if "forged" in filename or "fake" in filename:
        final_status = MANUAL_REVIEW_FLAG
        notes = "Heuristic match: flagged for manual review"
    else:
        # small chance to flag ambiguous files
        if uuid.uuid4().int % 10 == 0:
            final_status = MANUAL_REVIEW_FLAG
            notes = "Randomly flagged for manual review"
        else:
            final_status = VERIFIED_FLAG
            notes = "Auto-verified by model (simulated)"

    # Update DB
    update_submission_status(submission_id, final_status, processing_time_sec=proc_time, notes=notes)


def start_worker_thread(submission_id: str, filepath: str):
    t = threading.Thread(target=simulated_verify_worker, args=(submission_id, filepath), daemon=True)
    t.start()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Credential Fraud Detection", layout="wide")
init_db()

# Session: track logged-in user id
if "user" not in st.session_state:
    st.session_state.user = None


def show_welcome():
    st.title("🎓 AI-powered Credential Fraud Detection")
    st.markdown(
        """
        Welcome! This demo front-end is built with **Streamlit** for uploading and verifying digital credentials (certificates).
        - Signup/Login to get started.
        - Upload a certificate (image/pdf) to run verification.
        - Monitor real-time processing status and system metrics in the dashboard.

        > Note: This frontend simulates verification. Replace the `simulated_verify_worker` function with your real inference call.
        """
    )
    st.info("For real deployment, secure authentication & HTTPS are required. This demo uses a local SQLite DB.")


def show_auth():
    st.header("🔐 Login / Signup")
    tabs = st.tabs(["Login", "Signup"])
    with tabs[0]:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                st.success(f"Logged in as {username}")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

    with tabs[1]:
        st.subheader("Signup")
        new_user = st.text_input("Choose username", key="signup_user")
        new_pass = st.text_input("Choose password", type="password", key="signup_pass")
        if st.button("Create account"):
            if not new_user or not new_pass:
                st.error("Username and password required")
            else:
                ok = create_user(new_user.strip(), new_pass)
                if ok:
                    st.success("Account created — please login")
                else:
                    st.error("Username already exists")


def show_upload_page():
    st.header("📤 Upload Certificate for Verification")
    st.write("Upload an image (jpg/png) or PDF of the certificate. The system will queue it for verification.")
    if st.session_state.user is None:
        st.warning("Please login first to upload documents.")
        return

    uploaded_file = st.file_uploader("Choose a certificate file", type=["jpg", "jpeg", "png", "pdf"])
    notes = st.text_area("Optional notes (visible to reviewer)", height=80)
    if uploaded_file is not None:
        # Save file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{Path(uploaded_file.name).stem}_{timestamp}{Path(uploaded_file.name).suffix}"
        filepath = UPLOADS_DIR / safe_name
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved upload: {safe_name}")

        # Add DB entry & start worker
        submission_id = add_submission(st.session_state.user["id"], safe_name, str(filepath))
        start_worker_thread(submission_id, str(filepath))
        st.info("Document queued for verification. Go to Dashboard to see status.")

    st.markdown("---")
    st.subheader("Your recent submissions")
    if st.session_state.user:
        subs = get_user_submissions(st.session_state.user["id"])
        if not subs:
            st.info("No submissions yet.")
        else:
            for s in subs[:10]:
                status = s["status"]
                uploaded_at = s["uploaded_at"]
                processing = s["processing_time_sec"] or 0.0
                time_saved = s["time_saved_sec"] or 0.0
                cols = st.columns([1, 2, 1, 1, 1])
                with cols[0]:
                    st.text(s["filename"])
                with cols[1]:
                    st.text(f"Uploaded: {uploaded_at}")
                with cols[2]:
                    st.text(f"Status: {status}")
                with cols[3]:
                    st.text(f"Processing: {processing:.1f}s")
                with cols[4]:
                    st.text(f"Saved: {time_saved / 60:.2f}m")
            st.markdown("Click **Dashboard** to see system-level metrics and queue.")


def show_dashboard():
    st.header("📈 Real-time Status Dashboard")
    st.write("System-level stats and verification queue.")
    auto_refresh = st.checkbox("Auto-refresh (page reload)", value=False)

    # Provide a manual refresh button; page reload shows updates
    if st.button("Refresh"):
        st.experimental_rerun()

    if auto_refresh:
        # Simple auto-refresh by re-running every few seconds. This uses a small sleep to throttle.
        # NOTE: Streamlit doesn't allow infinite loops in the main script; this pattern forces a re-run.
        time.sleep(2)
        st.experimental_rerun()

    stats = get_system_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", stats["total"])
    col2.metric("Verified", stats["verified"])
    col3.metric("Flagged - Manual Review", stats["manual"])
    col4.metric("Processing Now", stats["processing"])

    st.metric("Estimated Time Saved (total)", f"{stats['total_time_saved_sec'] / 60:.1f} minutes")

    st.markdown("---")
    st.subheader("Pending / Recent Submissions (most recent first)")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM submissions ORDER BY uploaded_at DESC LIMIT 50")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        st.info("No submissions yet.")
        return

    for r in rows:
        r = dict(r)
        with st.expander(f"{r['filename']} — {r['status']} (uploaded {r['uploaded_at']})", expanded=False):
            cols = st.columns([1, 1, 1, 1, 1])
            cols[0].write(f"Filename: {r['filename']}")
            cols[1].write(f"Status: **{r['status']}**")
            cols[2].write(f"Processing: {r['processing_time_sec']:.1f}s")
            cols[3].write(f"Time Saved: {r['time_saved_sec'] / 60:.2f} minutes")
            cols[4].write(f"Notes: {r['verifier_notes'] or '—'}")
            # Show thumbnail if image
            try:
                if Path(r['filepath']).exists() and Path(r['filepath']).suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    img = Image.open(r['filepath'])
                    st.image(img, width=300)
            except Exception:
                st.write("Could not load preview.")

            # Allow admin-like manual override for demo (in production, protect this)
            st.markdown("**Admin tools (demo):**")
            c1, c2, c3 = st.columns(3)
            if c1.button("Mark Verified", key=f"verify_{r['id']}"):
                update_submission_status(r['id'], VERIFIED_FLAG, processing_time_sec=r['processing_time_sec'] or 0,
                                         notes="Manually verified via dashboard")
                st.success("Marked verified")
                st.experimental_rerun()
            if c2.button("Flag Manual Review", key=f"flag_{r['id']}"):
                update_submission_status(r['id'], MANUAL_REVIEW_FLAG, processing_time_sec=r['processing_time_sec'] or 0,
                                         notes="Manually flagged via dashboard")
                st.warning("Flagged for manual review")
                st.experimental_rerun()
            if c3.button("Retry (simulate)", key=f"retry_{r['id']}"):
                update_submission_status(r['id'], PROCESSING_FLAG, processing_time_sec=0, notes="Retry requested")
                start_worker_thread(r['id'], r['filepath'])
                st.info("Retry started")
                st.experimental_rerun()


# ----------------------------
# Router / Navigation
# ----------------------------
menu = ["Welcome", "Login / Signup", "Upload Certificate", "Dashboard", "Logout"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Welcome":
    show_welcome()
elif choice == "Login / Signup":
    show_auth()
elif choice == "Upload Certificate":
    show_upload_page()
elif choice == "Dashboard":
    show_dashboard()
elif choice == "Logout":
    st.session_state.user = None
    st.success("Logged out")
    st.experimental_rerun()
