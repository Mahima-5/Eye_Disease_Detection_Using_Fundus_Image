"""
FastAPI Backend — VisionaryAI Eye Disease Detection
=====================================================
Full DB layer (SQLite) + ML inference + Grad-CAM
Run: uvicorn app:app --reload --host 127.0.0.1 --port 8000
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import time
import sqlite3
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from typing import Optional
import base64

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE         = 300
MODEL_PATH       = "best_model.keras"
CLASS_NAMES_PATH = "class_names.json"
DB_PATH          = "visionary.db"

model            = None
class_names      = []
LAST_CONV_NAME   = None
conv_owner       = None


def find_last_conv_layer(m):
    result = None
    for layer in m.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            result = (m, layer)
        elif hasattr(layer, 'layers'):
            sub = find_last_conv_layer(layer)
            if sub is not None:
                result = sub
    return result


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = get_db()
    con.executescript("""
        PRAGMA journal_mode = WAL;

        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            age         INTEGER,
            gender      TEXT,
            phone       TEXT DEFAULT '',
            email       TEXT DEFAULT '',
            address     TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
            scan_date       TEXT DEFAULT (datetime('now')),
            eye_side        TEXT DEFAULT 'Right',
            prediction      TEXT,
            confidence      REAL,
            top3_json       TEXT,
            gradcam_b64     TEXT,
            image_path      TEXT DEFAULT '',
            inference_ms    REAL DEFAULT 0,
            notes           TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS prescriptions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id        INTEGER REFERENCES scans(id) ON DELETE SET NULL,
            patient_id     INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
            symptoms       TEXT DEFAULT '',
            medications    TEXT DEFAULT '',
            investigations TEXT DEFAULT '',
            follow_up      TEXT DEFAULT '',
            referral       TEXT DEFAULT 'None',
            advice         TEXT DEFAULT '',
            rx_number      TEXT UNIQUE,
            created_at     TEXT DEFAULT (datetime('now'))
        );
    """)
    con.commit()
    con.close()
    print("✅ Database ready:", DB_PATH)


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class PatientCreate(BaseModel):
    patient_id: Optional[str] = None
    name: str
    age: Optional[int] = None
    gender: Optional[str] = "Male"
    phone: Optional[str] = ""
    email: Optional[str] = ""
    address: Optional[str] = ""


class PatientUpdate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = ""
    email: Optional[str] = ""
    address: Optional[str] = ""


class PrescriptionCreate(BaseModel):
    patient_id: int
    scan_id: Optional[int] = None
    symptoms: Optional[str] = ""
    medications: Optional[str] = ""
    investigations: Optional[str] = ""
    follow_up: Optional[str] = ""
    referral: Optional[str] = "None"
    advice: Optional[str] = ""


class PrescriptionUpdate(BaseModel):
    symptoms: Optional[str] = ""
    medications: Optional[str] = ""
    investigations: Optional[str] = ""
    follow_up: Optional[str] = ""
    referral: Optional[str] = "None"
    advice: Optional[str] = ""


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, LAST_CONV_NAME, conv_owner

    init_db()

    print("Loading ML model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ Model loaded — {len(class_names)} classes")

        found = find_last_conv_layer(model)
        if found:
            conv_owner, conv_layer = found
            LAST_CONV_NAME = conv_layer.name
            print(f"✅ Grad-CAM layer: '{LAST_CONV_NAME}'")
        else:
            print("⚠️  No Conv2D — Grad-CAM disabled")
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}. Inference will be unavailable.")

    yield
    print("Shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="VisionaryAI — Eye Disease Detection API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ML Helpers
# ─────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def generate_gradcam_b64(image_bytes: bytes, pred_idx: int) -> str:
    if model is None or LAST_CONV_NAME is None or conv_owner is None:
        return ""
    try:
        img_array = preprocess(image_bytes)
        img_tensor = tf.constant(img_array, dtype=tf.float32)

        conv_layer = conv_owner.get_layer(LAST_CONV_NAME)
        feature_map_model = tf.keras.Model(
            inputs=conv_owner.input,
            outputs=conv_layer.output,
        )

        with tf.GradientTape() as tape:
            feature_maps = feature_map_model(img_tensor, training=False)
            tape.watch(feature_maps)
            predictions = model(img_tensor, training=False)
            class_score = predictions[:, pred_idx]

        grads = tape.gradient(class_score, feature_maps)
        if grads is None:
            return ""

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        feature_maps_np = feature_maps[0].numpy()
        weighted = feature_maps_np * pooled_grads[np.newaxis, np.newaxis, :]
        heatmap = np.mean(weighted, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 1e-8:
            heatmap /= heatmap.max()

        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        original = np.array(
            Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        )
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_colored, 0.45, 0)

        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf).decode("utf-8")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ""


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gradcam_layer": LAST_CONV_NAME,
        "num_classes": len(class_names),
    }


@app.get("/classes")
def get_classes():
    return {"total": len(class_names), "classes": class_names}


# ─────────────────────────────────────────────
# ML Inference
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    start = time.time()
    arr = preprocess(image_bytes)
    preds = model.predict(arr, verbose=0)[0]
    elapsed = round((time.time() - start) * 1000, 1)

    top3_idx = np.argsort(preds)[::-1][:3]
    top1_idx = int(top3_idx[0])

    return JSONResponse({
        "prediction": class_names[top1_idx],
        "confidence": round(float(preds[top1_idx]) * 100, 2),
        "inference_time_ms": elapsed,
        "top3_predictions": [
            {
                "rank": i + 1,
                "class": class_names[int(top3_idx[i])],
                "confidence_percent": round(float(preds[top3_idx[i]]) * 100, 2),
            }
            for i in range(3)
        ],
        "gradcam_overlay_base64": generate_gradcam_b64(image_bytes, top1_idx),
    })


@app.post("/predict/{patient_id}")
async def predict_and_save(patient_id: int, eye_side: str = "Right", file: UploadFile = File(...)):
    """Predict and automatically save scan to database."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    image_bytes = await file.read()
    start = time.time()
    arr = preprocess(image_bytes)
    preds = model.predict(arr, verbose=0)[0]
    elapsed = round((time.time() - start) * 1000, 1)

    top3_idx = np.argsort(preds)[::-1][:3]
    top1_idx = int(top3_idx[0])
    gradcam_b64 = generate_gradcam_b64(image_bytes, top1_idx)

    top3 = [
        {"rank": i+1, "class": class_names[int(top3_idx[i])], "confidence_percent": round(float(preds[top3_idx[i]])*100, 2)}
        for i in range(3)
    ]

    con = get_db()
    try:
        cur = con.execute("""
            INSERT INTO scans (patient_id, eye_side, prediction, confidence, top3_json, gradcam_b64, inference_ms, scan_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id, eye_side,
            class_names[top1_idx],
            round(float(preds[top1_idx]) * 100, 2),
            json.dumps(top3), gradcam_b64,
            elapsed, datetime.now().isoformat()
        ))
        scan_id = cur.lastrowid
        con.commit()
    finally:
        con.close()

    return JSONResponse({
        "scan_id": scan_id,
        "prediction": class_names[top1_idx],
        "confidence": round(float(preds[top1_idx]) * 100, 2),
        "inference_time_ms": elapsed,
        "top3_predictions": top3,
        "gradcam_overlay_base64": gradcam_b64,
    })


# ─────────────────────────────────────────────
# Patients CRUD
# ─────────────────────────────────────────────
@app.get("/patients")
def list_patients():
    con = get_db()
    rows = con.execute("""
        SELECT p.*, COUNT(s.id) as scan_count, MAX(s.scan_date) as last_scan,
               (SELECT prediction FROM scans WHERE patient_id = p.id ORDER BY scan_date DESC LIMIT 1) as last_diagnosis
        FROM patients p LEFT JOIN scans s ON s.patient_id = p.id
        GROUP BY p.id ORDER BY p.created_at DESC
    """).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int):
    con = get_db()
    row = con.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Patient not found")
    return dict(row)


@app.post("/patients")
def create_patient(data: PatientCreate):
    pid = data.patient_id or f"VIS-{int(time.time() * 1000) % 1000000:06d}"
    con = get_db()
    try:
        cur = con.execute("""
            INSERT INTO patients (patient_id, name, age, gender, phone, email, address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pid, data.name, data.age, data.gender, data.phone, data.email, data.address))
        con.commit()
        return {"id": cur.lastrowid, "patient_id": pid}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Patient ID already exists")
    finally:
        con.close()


@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, data: PatientUpdate):
    con = get_db()
    con.execute("""
        UPDATE patients SET name=?, age=?, gender=?, phone=?, email=?, address=?
        WHERE id=?
    """, (data.name, data.age, data.gender, data.phone, data.email, data.address, patient_id))
    con.commit()
    con.close()
    return {"success": True}


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int):
    con = get_db()
    con.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    con.commit()
    con.close()
    return {"success": True}


# ─────────────────────────────────────────────
# Scans CRUD
# ─────────────────────────────────────────────
@app.get("/patients/{patient_id}/scans")
def list_patient_scans(patient_id: int):
    con = get_db()
    rows = con.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as pid
        FROM scans s JOIN patients p ON p.id = s.patient_id
        WHERE s.patient_id = ? ORDER BY s.scan_date DESC
    """, (patient_id,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/scans")
def list_all_scans(limit: int = 50):
    con = get_db()
    rows = con.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as pid
        FROM scans s JOIN patients p ON p.id = s.patient_id
        ORDER BY s.scan_date DESC LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/scans/{scan_id}")
def get_scan(scan_id: int):
    con = get_db()
    row = con.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as pid, p.age, p.gender
        FROM scans s JOIN patients p ON p.id = s.patient_id
        WHERE s.id = ?
    """, (scan_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Scan not found")
    return dict(row)


# ─────────────────────────────────────────────
# Prescriptions CRUD
# ─────────────────────────────────────────────
@app.get("/prescriptions")
def list_prescriptions(limit: int = 50):
    con = get_db()
    rows = con.execute("""
        SELECT rx.*, p.name as patient_name, p.age, p.gender
        FROM prescriptions rx JOIN patients p ON p.id = rx.patient_id
        ORDER BY rx.created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/patients/{patient_id}/prescriptions")
def list_patient_prescriptions(patient_id: int):
    con = get_db()
    rows = con.execute("""
        SELECT rx.*, p.name as patient_name
        FROM prescriptions rx JOIN patients p ON p.id = rx.patient_id
        WHERE rx.patient_id = ? ORDER BY rx.created_at DESC
    """, (patient_id,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/prescriptions")
def create_prescription(data: PrescriptionCreate):
    rx_num = f"RX-{datetime.now().year}-{int(time.time() * 1000) % 10000:04d}"
    con = get_db()
    try:
        cur = con.execute("""
            INSERT INTO prescriptions
                (scan_id, patient_id, symptoms, medications, investigations, follow_up, referral, advice, rx_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data.scan_id, data.patient_id, data.symptoms, data.medications,
              data.investigations, data.follow_up, data.referral, data.advice, rx_num))
        con.commit()
        return {"id": cur.lastrowid, "rx_number": rx_num}
    finally:
        con.close()


@app.put("/prescriptions/{rx_id}")
def update_prescription(rx_id: int, data: PrescriptionUpdate):
    con = get_db()
    con.execute("""
        UPDATE prescriptions SET symptoms=?, medications=?, investigations=?,
        follow_up=?, referral=?, advice=? WHERE id=?
    """, (data.symptoms, data.medications, data.investigations,
          data.follow_up, data.referral, data.advice, rx_id))
    con.commit()
    con.close()
    return {"success": True}


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    con = get_db()
    total_patients = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_scans    = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    today_scans    = con.execute("SELECT COUNT(*) FROM scans WHERE date(scan_date) = date('now')").fetchone()[0]
    total_rx       = con.execute("SELECT COUNT(*) FROM prescriptions").fetchone()[0]
    recent = con.execute("""
        SELECT s.scan_date, s.prediction, s.confidence, p.name as patient_name
        FROM scans s JOIN patients p ON p.id = s.patient_id
        ORDER BY s.scan_date DESC LIMIT 5
    """).fetchall()
    con.close()
    return {
        "totalPatients": total_patients,
        "totalScans": total_scans,
        "todayScans": today_scans,
        "totalRx": total_rx,
        "recentActivity": [dict(r) for r in recent],
    }
