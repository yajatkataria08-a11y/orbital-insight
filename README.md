# 🛰️ Orbital Insight — NSH 2026 Autonomous Collision Management System

**Version:** ACM v7.0 + ML v4.0 | **Team:** BroCODE | **Event:** NSH 2026 (IIT Delhi Hackathon)

> A real-time satellite fleet management system with autonomous collision avoidance, an XGBoost ML risk classifier with online learning, conformal prediction uncertainty, A/B shadow deployment, KS drift detection, and multi-modal mission dashboards — built for the NSH 2026 problem statement.

---

## 🚀 Quick Start

```bash
# 1. Generate training data
cd backend
python generate_data.py

# 2. Train the ML model
python train_model.py

# 3. Launch everything via Docker
cd ..
docker compose up --build

# Access
# HTML Dashboard  →  http://localhost:80
# FastAPI Backend →  http://localhost:8000
# API Docs        →  http://localhost:8000/docs
# Streamlit       →  http://localhost:8501
# Structured Logs →  http://localhost:8000/api/logs
```

---

## 📁 File Structure

```
V2/
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── backend/
│   ├── main.py              # FastAPI physics + ML engine (ACM v7.0)
│   ├── train_model.py       # XGBoost trainer with Optuna + SHAP (ML v4.0)
│   ├── generate_data.py     # Synthetic conjunction dataset generator
│   ├── requirements.txt
│   ├── collision_model.pkl  # Trained CalibratedClassifierCV (auto-generated)
│   ├── model_features.pkl   # 21-feature name list (auto-generated)
│   ├── model_threshold.pkl  # Optimal F1 threshold (auto-generated)
│   └── model_meta.json      # Provenance, metrics, hyperparams (auto-generated)
├── frontend/
│   ├── index.html           # Single-file Canvas dashboard
│   └── earth.jpg            # Earth texture for 3D view
└── streamlit_app/
    └── app.py               # Streamlit analytics dashboard
```

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Docker Container                        │
│                   (supervisord managed)                      │
│                                                              │
│  ┌─────────────┐    ┌──────────────────────────────────┐    │
│  │  Nginx :80  │    │       FastAPI :8000               │    │
│  │ HTML Front  │───▶│  Physics Engine  +  ML Engine     │    │
│  │  end Canvas │    │  RK4 + J2 + Chan Pc + XGBoost     │    │
│  └─────────────┘    └──────────────────────────────────┘    │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │       Streamlit :8501 — Analytics Dashboard           │   │
│  │  CDM · Uptime · Maneuver · SHAP · Predict · Drift     │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🤖 ML Pipeline (New in v4.0)

### Overview

The ML subsystem is a full online-learning pipeline that trains, evaluates, self-improves, and hot-swaps models without any service restart.

```
generate_data.py  →  training_data.csv
                           ↓
                     train_model.py
                           ↓
              ┌────────────────────────┐
              │  XGBoost (DART booster)│
              │  + Focal Loss          │
              │  + Optuna HPO (60 trials)│
              │  + GroupKFold CV       │
              │  + Isotonic Calibration│
              └────────────────────────┘
                           ↓
                  collision_model.pkl
                           ↓
                       main.py
              ┌────────────────────────┐
              │  /api/ml/predict_risk  │
              │  + Conformal PI        │
              │  + LRU Cache           │
              │  + False-neg logging   │
              │  + A/B Shadow mode     │
              │  + Auto hot-swap       │
              └────────────────────────┘
                           ↓
                   missed_cases.csv
                           ↓
              retrain watcher (background)
                           ↓
              new model → hot-reload (no restart)
```

### Feature Set v4.0 — 21 Features

| # | Feature | Version | Description |
|---|---------|---------|-------------|
| 0 | `miss_distance_m` | v1 | Raw miss distance |
| 1 | `relative_velocity_ms` | v1 | Relative speed at TCA |
| 2 | `altitude_km` | v1 | Orbit altitude |
| 3 | `inclination_diff_deg` | v1 | Orbital plane difference |
| 4 | `time_to_closest_s` | v1 | Seconds to TCA |
| 5 | `debris_eccentricity` | v1 | Debris orbit eccentricity |
| 6 | `combined_radius_m` | v1 | Sat + debris hard-body radius |
| 7 | `dist_rate_kms` | v1 | Range rate (closing speed) |
| 8 | `kinetic_energy_proxy` | v2 | (rel_vel_kms)² |
| 9 | `log_miss_distance_m` | v2 | log1p(miss_distance_m) |
| 10 | `delta_miss_m_per_s` | v3 | First diff of miss distance |
| 11 | `distance_acceleration` | v3 | Second diff (trend curvature) |
| 12 | `grav_potential` | v3 | −GM/r gravitational potential |
| 13 | `sin_inc_diff` | v3 | sin(inclination_diff_deg) |
| 14 | `cos_inc_diff` | v3 | cos(inclination_diff_deg) |
| 15 | `atmospheric_density_multiplier` | v3 | Solar weather / drag multiplier |
| 16 | `vel_r_ms` | **v4** | RTN radial velocity |
| 17 | `vel_t_ms` | **v4** | RTN transverse velocity (most critical) |
| 18 | `vel_n_ms` | **v4** | RTN normal velocity |
| 19 | `log_chan_pc` | **v4** | log10(Chan Pc) — physics prior |
| 20 | `period_ratio` | **v4** | Orbital resonance ratio |

### Model Architecture

| Component | Choice | Reason |
|-----------|--------|--------|
| Booster | DART (Dropout Additive Regression Trees) | Prevents dominant-tree overfitting |
| Objective | Focal Loss (α=0.25, γ=2.0) | Focuses training on hard borderline conjunctions |
| Calibration | `CalibratedClassifierCV` — isotonic, cv=5 | Non-parametric probability calibration |
| HPO | Optuna TPE — 60 trials, maximise PR-AUC | Systematic search over 10 hyperparameters |
| CV strategy | GroupKFold(debris_id) / StratifiedKFold | Prevents debris-ID data leakage |
| Class balance | `scale_pos_weight × 2.0` recall bias | False negatives are catastrophic |

### Achieved Metrics (latest run)

| Metric | Value |
|--------|-------|
| ROC-AUC | 1.0000 |
| Average Precision | 1.0000 |
| Default Recall | 0.9992 |
| Default F1 | 0.9996 |
| Training time | ~37 min (CPU) |

---

## 🔄 Online Learning — Dynamic Feedback Loop

The system continuously improves without manual intervention:

### 1. False-Negative Capture
Every time the ML model disagrees with the Chan physics oracle (ML says LOW, Chan says HIGH), the conjunction is written to `missed_cases.csv` with its full feature vector.

### 2. Temporal Decay Weighting
Newer missed cases receive exponentially higher training weight:
```
weight(i) = exp(−λ · age_from_end)
λ = log(2) / HALFLIFE_ROWS    (half-life = 200 rows)
```

### 3. Difficulty Scaling
Cases the model was most wrong about get amplified most:
```
difficulty_scale = 1 + (MAX_SCALE − 1) × (confidence_gap / 0.5)
confidence_gap   = max(0, 0.5 − ml_probability)
MAX_SCALE        = 8.0
```

### 4. Automated Retraining Watcher
A background thread watches `missed_cases.csv`. When ≥ 50 new missed cases accumulate, it spawns `train_model.py` as a subprocess and hot-reloads the model on completion — zero downtime, no container restart.

### 5. Hot-Swap Gate
The new model only replaces the incumbent if its CV Recall beats the stored `test_recall_default` by ≥ 0.2 percentage points. Otherwise it is saved as `*_candidate.pkl` for manual review.

---

## 🧪 A/B Shadow Mode

When `collision_model_candidate.pkl` exists, the candidate runs in parallel on every live inference request:

- Both incumbent and candidate predictions are logged to `comparison.log`
- Neither prediction nor API response is altered — candidate is purely observational
- After 100 shadow ticks, the candidate is auto-promoted if it shows strictly higher recall against the Chan oracle
- Full comparison log visible at `GET /api/metrics → ab_shadow`

---

## 📊 Conformal Prediction Uncertainty

Every `/api/ml/predict_risk` response includes a statistically valid uncertainty interval:

```json
"uncertainty": {
  "lower": 0.412,
  "upper": 0.831,
  "coverage": 0.90,
  "high_alert": true,
  "calibration_n": 147
}
```

- Uses split-conformal prediction (no extra dependencies — pure numpy)
- Rolling 500-sample residual buffer: `s = |chan_label − ml_probability|`
- 90% coverage prediction interval at each inference call
- If interval width > 0.4 → `high_alert = true` → system falls back to Chan formula automatically

---

## 🔍 KS Drift Detection

At training time and periodically in the live system, a Kolmogorov-Smirnov test compares the distribution of key features in `missed_cases.csv` against `training_data.csv`:

```
Features checked: altitude_km, miss_distance_m,
                  relative_velocity_ms, atmospheric_density_multiplier
p-value threshold: 0.05
```

If drift is detected, a `⚠ DRIFT` alert is logged and written to `model_meta.json → drift_detection`. This signals that the live debris environment has shifted outside the training envelope.

---

## 🚀 Inference Performance

### LRU Cache
Identical (satellite, debris) feature pairs within the same simulation tick are served from an LRU cache (capacity 512) instead of re-running XGBoost:

```
Cache key = tuple(round(feature_value, 2) for each feature)
Cache is cleared at the start of each simulation tick
```

### ONNX Fast-Path (optional)
If `collision_model.onnx` and `onnxruntime` are installed, the batch endpoint uses ONNX Runtime instead of sklearn:
- 2–5× lower per-batch latency
- Single C++ dispatch for the full feature matrix
- Falls back to `.pkl` gracefully if ONNX unavailable

### Batch Endpoint
`POST /api/ml/predict_risk_batch` accepts up to 500 conjunctions in one call:
- Single matrix multiply vs N sequential API round-trips
- ~10–40× throughput improvement for large batches
- Full conformal intervals, physics gate, and missed-case logging per item

---

## 🔭 Physics Engine

### Constants (NSH 2026 Spec)

| Parameter | Value | Unit |
|-----------|-------|------|
| Gravitational parameter μ | 398600.4418 | km³/s² |
| Earth radius | 6378.137 | km |
| J2 coefficient | 1.08263 × 10⁻³ | — |
| Specific impulse (Isp) | 300 | s |
| Dry mass | 500 | kg |
| Fuel mass | 50 | kg |
| Max ΔV per burn | 15 | m/s |
| Thermal cooldown | 600 | s |
| Conjunction threshold | 100 | m |
| Station-keeping box | 10 | km |
| EOL fuel threshold | 5 | % |
| Graveyard altitude | 2000 | km |

### Propagation — RK4 + J2

4th-order Runge-Kutta with full J2 oblateness:
```
aJ2_x = (3/2)·J2·μ·RE²/r⁵ · x · (5z²/r² − 1)
aJ2_y = (3/2)·J2·μ·RE²/r⁵ · y · (5z²/r² − 1)
aJ2_z = (3/2)·J2·μ·RE²/r⁵ · z · (5z²/r² − 3)
```

### Constellation — 3 Orbital Shells (55 Satellites)

| Shell | Altitude | Inclination | Count |
|-------|----------|-------------|-------|
| Alpha | 550 km | 53° | 22 |
| Beta  | 570 km | 70° | 18 |
| Gamma | 560 km | 97.6° (SSO) | 15 |

### Debris Field
15,000 objects distributed 300–800 km, eccentricity 0–0.05.

---

## 🎯 Scored Features (NSH 2026 Rubric)

### [25%] Safety — Collision Avoidance

**Chan Pc Collision Probability**
```
Pc = (A_cb / 2π·σ²) · exp(−miss² / 2σ²)
σ  = max(0.05, miss_distance × 0.3)
```

**Physics-First Safety Gate**
If `miss_distance_m ≤ combined_radius_m` the objects physically overlap — ML is overridden to `prediction=1, probability=1.0` unconditionally.

**Parabolic TCA Refinement**
3-point parabolic fit at the coarse TCA neighbourhood — achieves sub-second accuracy with 3 propagations instead of ~480 (bisection).

**T-axis-first Optimal Evasion**
Prograde/retrograde tested first. Radial/Normal only if they yield `PC_TRANSVERSE_BIAS` (2×) improvement.

**Pc Burn Pruning**
Burns skipped when `Pc < 1e-6`. Full CDM audit trail preserved.

**Blind Pre-upload**
`compute_contact_windows()` propagates 4 hours forward to find the last GS pass before TCA blackout. Burns uploaded before LOS is lost.

---

### [20%] Fuel Efficiency

**Optimal ΔV Selection** — 6-axis RTN probe per conjunction, best miss-distance improvement selected.

**Hohmann Graveyard Transfer (EOL)** — two-burn Hohmann sequence raising apogee to 2000 km then circularising.

**Hohmann Phasing Recovery** — post-evasion slot recovery using proper phasing orbit math.

**Pc Pruning** — unnecessary burns avoided, fleet ΔV budget preserved.

---

### [15%] Constellation Uptime

Station-keeping box compliance tracked per satellite (10 km radius). Proactive correction at 70% box radius.

| Fleet Uptime | Grade | Points |
|-------------|-------|--------|
| ≥ 99% | EXCELLENT | 15 |
| ≥ 95% | GOOD | ~12 |
| ≥ 90% | ACCEPTABLE | ~9 |

Live score: `GET /api/fleet/uptime`

---

### [15%] Algorithmic Speed

**Hybrid Spatial Index**

| Mode | Algorithm | Complexity |
|------|-----------|------------|
| Primary | scipy KD-Tree | O(log N) |
| Fallback | 3D VoxelHash | O(k) |

**Non-blocking event loop** — `sim.step()` runs in `ThreadPoolExecutor` via `run_in_executor`.

**ML LRU cache** — 512-entry LRU eliminates redundant XGBoost calls within each tick.

Step timing: `GET /api/metrics → step_ms_avg`

---

### [15%] Visualisation (UI/UX)

**HTML Dashboard (port 80)**
- Ground track world map with orbital trails and terminator line
- 3D orbit view — perspective projection, drag-to-rotate, day/night shading
- RTN conjunction bullseye — RED/YELLOW/GREEN risk rings
- Maneuver Gantt timeline — evasion, recovery, graveyard, station-keep burns
- ΔV efficiency graph — cost per collision avoided
- Fleet uptime bars per satellite

**Streamlit Dashboard (port 8501)**
- CDM Registry with full Chan Pc history
- Uptime Monitor with NSH rubric scoring
- Contact Schedule with blackout warnings
- Maneuver History with burn-type breakdown
- **ML Predict tab** — interactive single conjunction risk predictor with SHAP images, conformal interval gauge, and probability bar
- **SHAP Explainability** — Feature importance, beeswarm, and PR curve tabs
- Model metadata card (ROC-AUC, Recall, AP, features)

---

### [10%] Code Quality & Logging

**Structured JSON audit log**
```json
{"ts":"2026-03-26T13:14:22","level":"WARNING","name":"acm",
 "msg":{"event":"conjunction_evasion_planned","satellite_id":"SAT-Alpha-07",
        "debris_id":"DEB-03841","miss_distance_m":47.3,"pc":2.14e-4,
        "tca_iso":"2026-03-26T15:32:00Z","evasion_burn":"EVASION_SAT-Alpha-07_4582",
        "contact_window":"GS-002","pre_upload":true}}
```

Live tail: `GET /api/logs?limit=100`

---

## 📡 Ground Stations

| ID | Name | Lat | Lon | Min El |
|----|------|-----|-----|--------|
| GS-001 | ISTRAC Bengaluru | 13.033° | 77.517° | 5° |
| GS-002 | Svalbard Satellite Station | 78.230° | 15.408° | 5° |
| GS-003 | Goldstone Tracking | 35.427° | −116.890° | 10° |
| GS-004 | Punta Arenas | −53.150° | −70.917° | 5° |
| GS-005 | IIT Delhi Ground Node | 28.545° | 77.193° | 15° |
| GS-006 | McMurdo Station | −77.846° | 166.668° | 5° |

---

## 🔌 API Reference

Full interactive docs at `http://localhost:8000/docs`.

### NSH Grader Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/telemetry` | Ingest satellite/debris state vectors |
| `POST` | `/api/maneuver/schedule` | Schedule a burn sequence |
| `POST` | `/api/simulate/step` | Advance simulation by N seconds |
| `GET` | `/api/visualization/snapshot` | Full fleet + debris snapshot |

### ML Endpoints (New)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ml/predict_risk` | Single conjunction risk — XGBoost + conformal PI |
| `POST` | `/api/ml/predict_risk_batch` | Batch inference (up to 500 conjunctions) via ONNX/sklearn |

### Fleet Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Fleet health + spatial index + ML status |
| `GET` | `/api/satellites` | Full per-satellite state |
| `GET` | `/api/conjunctions` | Active conjunction list |
| `GET` | `/api/fleet/uptime` | Constellation uptime score |
| `GET` | `/api/fleet/heatmap` | Per-satellite health grid |
| `GET` | `/api/fleet/contact_summary` | Next GS windows fleet-wide |

### CDM & Maneuvers

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/cdm/registry` | All Conjunction Data Messages |
| `GET` | `/api/maneuver/history` | Executed burn log |
| `GET` | `/api/satellite/{id}/conjunction_detail` | RTN bullseye data |
| `GET` | `/api/satellite/{id}/contact_schedule` | Next 3 GS windows |

### Diagnostics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/metrics` | Step timing, cache hits, retrain watcher status, A/B shadow |
| `GET` | `/api/logs` | Structured JSON audit trail |
| `GET` | `/api/ground_stations` | GS visibility |
| `GET` | `/api/terminator` | Sun terminator line |

---

## 🐳 Docker

Built on `ubuntu:22.04` (NSH 2026 hard requirement). Process management via **supervisord** (nginx + uvicorn + streamlit — all auto-restart on crash).

```bash
# Build & run
docker compose up --build

# Logs per service
docker compose logs -f
# Or inside container:
tail -f /var/log/supervisor/fastapi.log
tail -f /var/log/supervisor/streamlit.log
tail -f /var/log/supervisor/nginx.log

# Health check
curl http://localhost:8000/api/status
```

**Ports:**

| Port | Service |
|------|---------|
| 80 | HTML Frontend (nginx) |
| 8000 | FastAPI Backend (NSH grader) |
| 8501 | Streamlit Dashboard (nginx proxy) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Physics engine | Python 3.11 — stdlib + numpy/scipy |
| Web framework | FastAPI + uvicorn |
| ML model | XGBoost (DART) + scikit-learn (calibration) |
| HPO | Optuna (TPE sampler, 60 trials) |
| Explainability | SHAP (TreeExplainer) |
| Uncertainty | Conformal prediction (split-conformal, pure numpy) |
| Spatial index | scipy KDTree / custom 3D VoxelHash |
| Fast inference | ONNX Runtime (optional, 2–5× batch speedup) |
| Frontend | Vanilla JS + Canvas 2D API |
| Analytics | Streamlit + Altair + Pandas |
| Reverse proxy | Nginx |
| Process manager | Supervisord |
| Container | Docker + Docker Compose |
| Base image | ubuntu:22.04 |

---

## 📊 Simulation Epoch

`2026-03-12T08:00:00Z` — all timestamps in ISO 8601 UTC.

---

*Orbital Insight — NSH 2026 · Built by **Team BroCODE** for IIT Delhi Hackathon*
