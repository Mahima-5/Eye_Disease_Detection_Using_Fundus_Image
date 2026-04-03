# 👁️ VisionaryAI Desktop — Eye Disease Detection

A full-stack **Electron desktop application** for AI-powered fundus image analysis.  
Built for ophthalmologists with EfficientNetB3 + Grad-CAM ML backend.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Electron Desktop App                       │
│                                                             │
│  ┌──────────────────────────────┐  ┌─────────────────────┐ │
│  │     React Frontend           │  │  Electron Main       │ │
│  │  (Vite, port 5173 in dev)    │◄─►│  (Node.js process)  │ │
│  │                              │  │                     │ │
│  │  • Dashboard                 │  │  • SQLite DB        │ │
│  │  • New Scan (AI analysis)    │  │  • File dialogs     │ │
│  │  • Patients CRUD             │  │  • PDF export       │ │
│  │  • Scan History              │  │  • IPC bridge       │ │
│  │  • Prescriptions + PDF       │  └──────────┬──────────┘ │
│  └──────────────┬───────────────┘             │             │
│                 │ fetch HTTP                  │ spawns      │
│                 ▼                             ▼             │
│  ┌─────────────────────────────┐  ┌─────────────────────┐ │
│  │   FastAPI Backend           │  │  SQLite Database    │ │
│  │   (Python, port 8000)       │  │  visionary.db       │ │
│  │                             │  │                     │ │
│  │  POST /predict              │  │  • patients         │ │
│  │  POST /predict/{patient_id} │  │  • scans            │ │
│  │  GET  /patients             │  │  • prescriptions    │ │
│  │  POST /patients             │  │                     │ │
│  │  GET  /scans                │  └─────────────────────┘ │
│  │  POST /prescriptions        │                           │
│  │  GET  /stats                │                           │
│  │                             │                           │
│  │  ML: EfficientNetB3         │                           │
│  │  Explainability: Grad-CAM   │                           │
│  └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
visionary-ai-desktop/
├── electron/
│   ├── main.js          ← Electron main process (window, IPC, DB, Python spawn)
│   └── preload.js       ← Context bridge (exposes window.api to React)
│
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx           ← Root with screen routing
│       ├── styles/
│       │   └── globals.css
│       ├── components/
│       │   ├── Sidebar.jsx
│       │   ├── Topbar.jsx
│       │   └── UI.jsx        ← Card, Badge, Btn, ConfBar, Avatar, etc.
│       └── screens/
│           ├── Dashboard.jsx      ← Stats + recent activity
│           ├── NewScan.jsx        ← Image upload + AI + Grad-CAM results
│           ├── Patients.jsx       ← Full patient CRUD
│           ├── ScanHistory.jsx    ← Browse all scans + detail view
│           └── Prescriptions.jsx  ← Create/view Rx + live PDF preview
│
├── backend/
│   ├── app.py              ← FastAPI server (ML + SQLite DB layer)
│   ├── train_model.py      ← EfficientNetB3 training script
│   ├── class_names.json    ← 39 eye condition labels
│   ├── requirements.txt
│   └── best_model.keras    ← Trained model (place here after training)
│
├── package.json
├── setup.bat               ← Windows one-click setup
├── setup.sh                ← Mac/Linux one-click setup
└── README.md
```

---

## ⚙️ Prerequisites

| Tool | Version | Download |
|---|---|---|
| Node.js | ≥ 18 | https://nodejs.org |
| Python | ≥ 3.10 | https://python.org |
| npm | ≥ 9 | (bundled with Node) |

---

## 🚀 Quick Setup

### Windows
```batch
setup.bat
```

### macOS / Linux
```bash
chmod +x setup.sh && ./setup.sh
```

### Manual setup
```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt
cd ..

# 2. Install Node dependencies
npm install

# 3. Start in development mode
npm run dev
```

---

## 🧠 ML Model Setup

The app works without the model (for patient management), but AI inference requires `best_model.keras`.

### Option A — Use your trained model
Copy your `best_model.keras` into the `backend/` folder.

### Option B — Train from scratch
```bash
cd backend

# Update DATA_DIR in train_model.py to your dataset path
python train_model.py --data_dir ./dataset
```

Your dataset must be organized as:
```
dataset/
├── 0.0.Normal/
│   ├── img001.jpg
│   └── ...
├── 1.0.DR2/
│   └── ...
└── ... (39 folders total)
```

---

## 🌐 API Reference

The FastAPI backend runs at `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + model status |
| GET | `/classes` | List all 39 detectable conditions |
| POST | `/predict` | Run AI inference on fundus image |
| POST | `/predict/{patient_id}` | Predict + auto-save scan to DB |
| GET | `/patients` | List all patients |
| POST | `/patients` | Create new patient |
| GET | `/patients/{id}` | Get patient by ID |
| PUT | `/patients/{id}` | Update patient |
| DELETE | `/patients/{id}` | Delete patient + all records |
| GET | `/patients/{id}/scans` | Patient scan history |
| GET | `/scans` | All scans (limit 50) |
| GET | `/scans/{id}` | Single scan with Grad-CAM |
| GET | `/prescriptions` | All prescriptions |
| POST | `/prescriptions` | Create prescription |
| PUT | `/prescriptions/{id}` | Update prescription |
| GET | `/stats` | Dashboard statistics |

**Swagger UI:** http://localhost:8000/docs (when server is running)

---

## 🗄️ Database Schema

SQLite at `%AppData%/visionary-ai-desktop/visionary.db` (production) or `backend/visionary.db` (dev)

```sql
patients       -- id, patient_id, name, age, gender, phone, email, address, created_at
scans          -- id, patient_id, scan_date, eye_side, prediction, confidence, top3_json,
               --    gradcam_b64, image_path, inference_ms, notes
prescriptions  -- id, scan_id, patient_id, symptoms, medications, investigations,
               --    follow_up, referral, advice, rx_number, created_at
```

---

## 📦 Building for Distribution

```bash
# Package into installer (.exe / .dmg / .AppImage)
npm run build
```

Output in `dist/` folder.

> **Note:** For true single-file distribution, bundle the Python environment with PyInstaller:
> ```bash
> cd backend
> pip install pyinstaller
> pyinstaller --onefile --name visionary-backend app.py
> ```
> Then update `electron/main.js` to point to the `dist/visionary-backend` executable.

---

## 🧩 Key Features

| Feature | Implementation |
|---|---|
| **AI Scan** | EfficientNetB3 → 39-class softmax |
| **Grad-CAM** | Visual explainability overlay |
| **Top-3 predictions** | Confidence bars per condition |
| **Auto-generated AI Report** | Findings / Interpretation / Recommendation |
| **Patient CRUD** | SQLite via Electron IPC |
| **Scan history** | Linked to patients, stored with Grad-CAM |
| **Prescriptions** | Full Rx form + live PDF preview |
| **PDF export** | Download printable prescription |
| **Backend status** | Real-time AI online/offline indicator |
| **Drag & drop** | Native file drop for fundus images |

---

## 🩺 Detectable Conditions (39 classes)

Normal · Tessellated fundus · Large optic cup · DR1 · DR2 · DR3 · Possible glaucoma ·
Optic atrophy · Severe hypertensive retinopathy · Disc swelling · Dragged Disc ·
Congenital disc abnormality · Retinitis pigmentosa · Bietti crystalline dystrophy ·
Peripheral retinal degeneration · Myelinated nerve fiber · Vitreous particles ·
Fundus neoplasm · BRVO · CRVO · Massive hard exudates · Yellow-white spots ·
Cotton-wool spots · Vessel tortuosity · Chorioretinal atrophy · Preretinal hemorrhage ·
Fibrosis · Laser Spots · Silicon oil in eye · Blur fundus (no PDR) ·
Blur fundus (suspected PDR) · RAO · Rhegmatogenous RD · CSCR · VKH disease ·
Maculopathy · ERM · MH · Pathological myopia

---

## 🤝 Credits

- **Model:** EfficientNetB3 (ImageNet pretrained, fine-tuned)
- **Explainability:** Grad-CAM (Selvaraju et al.)
- **Desktop:** Electron + React + Vite
- **Backend:** FastAPI + Uvicorn
- **Database:** SQLite (via better-sqlite3 in Electron)
- **Project:** C-DAC Hackathon — VisionaryAI
