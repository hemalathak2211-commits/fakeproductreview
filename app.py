# ============================================================
#  app.py  —  Flask Backend with ADWIN Concept Drift Detection
#  Run:  python app.py
# ============================================================
#
#  SETUP:
#  pip install flask flask-cors scikit-learn pandas numpy river
#  python train_model.py   (run once to train model)
#  python app.py           (start server on port 5000)
#
# ============================================================

from flask        import Flask, request, jsonify
from flask_cors   import CORS
import pickle, os, sqlite3, datetime
import numpy  as np
import pandas as pd

# ── ADWIN Import ─────────────────────────────────────────────
try:
    from river import drift
    ADWIN_AVAILABLE = True
    print("✅ ADWIN (river) loaded successfully")
except ImportError:
    ADWIN_AVAILABLE = False
    print("⚠️  river not installed. Run: pip install river")

app  = Flask(__name__)
CORS(app)

DB_PATH         = "reviews.db"
MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ── ADWIN Instance ───────────────────────────────────────────
adwin          = drift.ADWIN() if ADWIN_AVAILABLE else None
drift_detected = False
error_count    = 0
total_count    = 0
retrain_count  = 0

# ── Create Database ──────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT    NOT NULL,
            result      TEXT    NOT NULL,
            confidence  REAL,
            drift_flag  INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Drift log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event       TEXT    NOT NULL,
            description TEXT,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Load Model ───────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH,      "rb") as f: model      = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
        print("✅ Model and vectorizer loaded")
        return model, vectorizer
    print("⚠️  model.pkl not found. Run train_model.py first")
    return None, None

model, vectorizer = load_model()

# ── Retrain Model ────────────────────────────────────────────
def retrain_model():
    global model, vectorizer, adwin, drift_detected, error_count
    print("🔄 Retraining model due to concept drift...")

    try:
        import re
        from sklearn.ensemble              import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection       import train_test_split

        # Load reviews from database
        conn  = sqlite3.connect(DB_PATH)
        rows  = conn.execute("SELECT text, result FROM reviews").fetchall()
        conn.close()

        if len(rows) < 20:
            print("⚠️  Not enough data to retrain (need 20+ reviews)")
            return False

        # Prepare data
        texts  = [r[0] for r in rows]
        labels = [r[1] for r in rows]

        def preprocess(t):
            t = str(t).lower()
            t = re.sub(r'[^a-z0-9\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        clean_texts = [preprocess(t) for t in texts]

        # Retrain
        new_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english', min_df=1)
        X = new_vectorizer.fit_transform(clean_texts)
        y = labels

        new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        new_model.fit(X, y)

        # Save new model
        with open(MODEL_PATH,      "wb") as f: pickle.dump(new_model,      f)
        with open(VECTORIZER_PATH, "wb") as f: pickle.dump(new_vectorizer, f)

        # Update global model
        model      = new_model
        vectorizer = new_vectorizer

        # Reset ADWIN
        adwin         = drift.ADWIN() if ADWIN_AVAILABLE else None
        drift_detected = False
        error_count    = 0

        # Log drift event
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO drift_log (event, description) VALUES (?, ?)",
            ("RETRAIN", "Model retrained after drift detected. Samples used: " + str(len(rows)))
        )
        conn.commit()
        conn.close()

        print("✅ Model retrained successfully with", len(rows), "samples")
        return True

    except Exception as e:
        print("❌ Retraining failed:", str(e))
        return False

# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "message":        "Fake Review Detection API is running!",
        "status":         "ok",
        "adwin_enabled":  ADWIN_AVAILABLE,
        "drift_detected": drift_detected,
        "total_analyzed": total_count,
        "retrain_count":  retrain_count
    })


@app.route("/predict", methods=["POST"])
def predict():
    global drift_detected, error_count, total_count, retrain_count, adwin

    data = request.get_json()
    if not data or "review" not in data:
        return jsonify({"error": "Missing review field"}), 400

    review_text = data["review"].strip()
    if not review_text:
        return jsonify({"error": "Review is empty"}), 400

    if model is None or vectorizer is None:
        return jsonify({"error": "Model not found. Run train_model.py first"}), 500

    # Step 1 — TF-IDF Vectorize
    import re
    def preprocess(t):
        t = str(t).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    features   = vectorizer.transform([preprocess(review_text)])

    # Step 2 — Random Forest Predict
    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    confidence = round(float(max(proba)) * 100, 1)

    # Step 3 — ADWIN Drift Detection
    total_count   += 1
    current_drift  = False

    if ADWIN_AVAILABLE and adwin is not None:
        # Use low confidence as error signal
        error = 1.0 if confidence < 70.0 else 0.0
        adwin.update(error)

        if adwin.drift_detected:
            drift_detected = True
            current_drift  = True
            error_count   += 1
            print("⚠️  ADWIN Drift Detected! Review #" + str(total_count))

            # Log drift
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO drift_log (event, description) VALUES (?, ?)",
                ("DRIFT", "Drift detected at review #" + str(total_count) + " confidence=" + str(confidence))
            )
            conn.commit()
            conn.close()

            # Auto retrain if enough data
            conn  = sqlite3.connect(DB_PATH)
            count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            conn.close()

            if count >= 20:
                success = retrain_model()
                if success:
                    retrain_count += 1

    # Step 4 — Save to Database
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO reviews (text, result, confidence, drift_flag) VALUES (?, ?, ?, ?)",
        (review_text, prediction, confidence, 1 if current_drift else 0)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "result":         prediction,
        "confidence":     confidence,
        "label":          "Genuine Review" if prediction == "Genuine" else "Fake Review",
        "drift_detected": current_drift,
        "adwin_enabled":  ADWIN_AVAILABLE,
        "total_analyzed": total_count
    })


@app.route("/results", methods=["GET"])
def get_results():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, text, result, confidence, drift_flag, created_at FROM reviews ORDER BY id DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return jsonify([
        {
            "id":         r[0],
            "text":       r[1],
            "result":     r[2],
            "confidence": r[3],
            "drift_flag": r[4],
            "date":       r[5]
        }
        for r in rows
    ])


@app.route("/stats", methods=["GET"])
def get_stats():
    conn    = sqlite3.connect(DB_PATH)
    total   = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    genuine = conn.execute("SELECT COUNT(*) FROM reviews WHERE result='Genuine'").fetchone()[0]
    fake    = conn.execute("SELECT COUNT(*) FROM reviews WHERE result='Fake'").fetchone()[0]
    drifts  = conn.execute("SELECT COUNT(*) FROM drift_log WHERE event='DRIFT'").fetchone()[0]
    retrains= conn.execute("SELECT COUNT(*) FROM drift_log WHERE event='RETRAIN'").fetchone()[0]
    conn.close()
    return jsonify({
        "total":          total,
        "genuine":        genuine,
        "fake":           fake,
        "drift_count":    drifts,
        "retrain_count":  retrains,
        "adwin_enabled":  ADWIN_AVAILABLE,
        "drift_detected": drift_detected
    })


@app.route("/drift-log", methods=["GET"])
def get_drift_log():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, event, description, created_at FROM drift_log ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([
        {
            "id":          r[0],
            "event":       r[1],
            "description": r[2],
            "date":        r[3]
        }
        for r in rows
    ])


@app.route("/delete/<int:review_id>", methods=["DELETE"])
def delete_review(review_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Review " + str(review_id) + " deleted"})


# ── Start Server ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Fake Review Detection API")
    print("  Running on: http://localhost:5000")
    print("  ADWIN Drift Detection:", "ENABLED" if ADWIN_AVAILABLE else "DISABLED")
    print("=" * 50)
    app.run(debug=True, port=5000)