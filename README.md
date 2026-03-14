# 🛰️ Orbital Insight — NSH 2026 Autonomous Collision Management System

**Version:** ACM v7.0 | **Team:** BroCODE | **Event:** NSH 2026 (IIT Delhi Hackathon)

> A real-time satellite fleet management system with autonomous collision avoidance, predictive contact scheduling, and multi-modal mission dashboards — built for the NSH 2026 problem statement.

---

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/YOURUSERNAME/orbital-insight.git
cd orbital-insight
docker compose up --build

# Access
# HTML Dashboard  →  http://localhost:80
# FastAPI Backend →  http://localhost:8000
# API Docs        →  http://localhost:8000/docs
# Streamlit       →  http://localhost:8501
# Structured Logs →  http://localhost:8000/api/logs
```

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
│                                                     │
│   ┌──────────────┐    ┌───────────────────────┐    │
│   │   Nginx :80  │    │   FastAPI :8000        │    │
│   │  HTML Front  │───▶│   Physics Engine       │    │
│   │  end (Canvas)│    │   RK4 + J2 + Chan Pc   │    │
│   └──────────────┘    └───────────────────────┘    │
│                                                     │
│   ┌──────────────────────────────────────────────┐  │
│   │   Streamlit :8501 — Analytics Dashboard      │  │
│   │   CDM Registry · Uptime Monitor · Maneuver   │  │
│   └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### File Structure

```
orbital-insight/
├── backend/
│   ├── main.py              # FastAPI physics engine (ACM v7.0)
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-file Canvas dashboard
├── streamlit_app/
│   └── app.py               # Streamlit analytics dashboard
├── start.sh                 # Process manager (nginx + uvicorn + streamlit)
├── Dockerfile               # Ubuntu 22.04 base
└── docker-compose.yml
```

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
| Comm latency | 10 | s |
| Conjunction threshold | 100 | m |
| Station-keeping box | 10 | km |
| EOL fuel threshold | 5 | % |
| Graveyard altitude | 2000 | km |

### Propagation — RK4 + J2

4th-order Runge-Kutta integrator with full J2 oblateness perturbation:

```
aJ2_x = (3/2) · J2 · μ · RE² / r⁵ · x · (5z²/r² − 1)
aJ2_y = (3/2) · J2 · μ · RE² / r⁵ · y · (5z²/r² − 1)
aJ2_z = (3/2) · J2 · μ · RE² / r⁵ · z · (5z²/r² − 3)
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
2D Gaussian approximation (Chan 1997):
```
Pc = (A_cb / 2π·σ²) · exp(−miss² / 2σ²)
σ = max(0.05, miss_distance × 0.3)
```

**Bisection TCA Refinement**
20-iteration bisection to sub-second Time of Closest Approach accuracy.

**T-axis-first Optimal Evasion**
Prograde/retrograde tested first. Radial/Normal only considered if they yield `PC_TRANSVERSE_BIAS` (2×) improvement — avoids expensive out-of-plane burns.

**Pc Burn Pruning**
Maneuvers skipped if `Pc < 1e-6`. Full CDM audit trail preserved with `pc_pruned=True`.

**Blind Pre-upload (§5.4 compliance)**
`compute_contact_windows()` propagates 4 hours forward to find the last ground station pass before TCA blackout. Burns are scheduled before LOS is lost.

---

### [20%] Fuel Efficiency

**Optimal ΔV Selection**
6-axis RTN probe at each conjunction. Best miss-distance improvement selected, scaled to actual required ΔV.

**Pc Pruning**
Unnecessary burns skipped — fleet ΔV budget preserved for critical avoidance.

**Hohmann Graveyard Transfer (EOL)**
Two-burn Hohmann sequence at end of life:
- Burn A: prograde at current alt → raises apogee to 2000 km
- Burn B: circularises at 2000 km graveyard orbit
- Fallback to single deorbit if fuel insufficient for full sequence

**Hohmann Phasing Recovery**
Post-evasion slot recovery using proper phasing orbit math — more fuel-efficient than fixed retrograde burns.

---

### [15%] Constellation Uptime

Station-keeping box compliance tracked per satellite (10 km radius). Proactive correction triggered at 70% box radius before violation.

**Scoring thresholds (NSH 2026 rubric):**

| Fleet Uptime | Grade | Points |
|-------------|-------|--------|
| ≥ 99% | EXCELLENT | 15 |
| ≥ 95% | GOOD | ~12 |
| ≥ 90% | ACCEPTABLE | ~9 |
| < 90% | POOR | — |

Live uptime score at `GET /api/fleet/uptime`.

---

### [15%] Algorithmic Speed

**Hybrid Spatial Index**

| Mode | Algorithm | Complexity |
|------|-----------|------------|
| Primary | scipy KD-Tree | O(log N) |
| Fallback | 3D VoxelHash (10km × 10° × 10°) | O(k) |

KD-Tree rebuilds every 60s against 15,000 debris. Active mode reported in `GET /api/status → spatial_index`.

**Event loop non-blocking**
`sim.step()` runs in a `ThreadPoolExecutor` thread via `run_in_executor`. API endpoints stay responsive even during heavy conjunction assessment.

Step timing exposed at `GET /api/metrics`:
```json
{
  "step_ms_avg": 12.4,
  "step_ms_max": 38.1,
  "spatial_index": "kdtree"
}
```

---

### [15%] Visualisation (UI/UX)

**HTML Dashboard (port 80)**
- Ground track world map with orbital trails and terminator line
- 3D orbit view — perspective projection, drag-to-rotate, day/night shading
- RTN conjunction bullseye — RED/YELLOW/GREEN risk rings
- Maneuver Gantt timeline — evasion, recovery, graveyard, station-keep burns
- ΔV efficiency graph — cost per collision avoided (m/s/avoidance ratio)
- Fleet uptime bars per satellite

**Streamlit Dashboard (port 8501)**
- CDM Registry with full Chan Pc history
- Uptime Monitor with NSH rubric scoring display
- Contact Schedule with blackout warnings
- Maneuver History with burn-type breakdown charts
- Ground Station network visibility

---

### [10%] Code Quality & Logging

**Structured JSON audit log (`acm.log`)**
Every significant decision emits a structured JSON line:
```json
{"ts":"2026-03-12T09:14:22","level":"WARNING","logger":"orbital_insight",
 "msg":"conjunction_evasion_planned","satellite_id":"SAT-Alpha-07",
 "debris_id":"DEB-03841","miss_distance_m":47.3,"pc":2.14e-4,
 "tca_iso":"2026-03-12T11:32:00.000Z","evasion_burn":"EVASION_SAT-Alpha-07_4582",
 "contact_window":"GS-002","pre_upload":true}
```

Viewable live at `GET /api/logs?limit=100`.

---

## 📡 Ground Stations (NSH 2026 §5.5.1)

| ID | Name | Lat | Lon | Min Elevation |
|----|------|-----|-----|--------------|
| GS-001 | ISTRAC Bengaluru | 13.033° | 77.517° | 5° |
| GS-002 | Svalbard Satellite Station | 78.230° | 15.408° | 5° |
| GS-003 | Goldstone Tracking | 35.427° | −116.890° | 10° |
| GS-004 | Punta Arenas | −53.150° | −70.917° | 5° |
| GS-005 | IIT Delhi Ground Node | 28.545° | 77.193° | 15° |
| GS-006 | McMurdo Station | −77.846° | 166.668° | 5° |

---

## 🔌 API Reference

All endpoints on `port 8000`. Full interactive docs at `/docs`.

### NSH Grader Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/telemetry` | Ingest satellite/debris state vectors |
| `POST` | `/api/maneuver/schedule` | Schedule a burn sequence |
| `POST` | `/api/simulate/step` | Advance simulation by N seconds |
| `GET` | `/api/visualization/snapshot` | Full fleet + debris snapshot |

### Fleet Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Fleet health + spatial index mode |
| `GET` | `/api/satellites` | Full per-satellite state |
| `GET` | `/api/conjunctions` | Active conjunction list |
| `GET` | `/api/fleet/uptime` | Constellation uptime score |
| `GET` | `/api/fleet/heatmap` | Per-satellite health grid |
| `GET` | `/api/fleet/contact_summary` | Next GS windows, fleet-wide |

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
| `GET` | `/api/metrics` | Step timing, index mode, counts |
| `GET` | `/api/logs` | Structured JSON audit trail |
| `GET` | `/api/ground_stations` | GS visibility |
| `GET` | `/api/terminator` | Sun terminator line |

---

## 🐳 Docker

Built on `ubuntu:22.04` (NSH 2026 hard requirement).

```bash
# Build
docker compose build --no-cache

# Run
docker compose up

# Logs
docker compose logs -f

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

## 🌐 Global Deployment (Railway)

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects Dockerfile
4. Add ports 8000, 80, 8501 under Settings → Networking
5. Set env var: `BACKEND_URL=http://localhost:8000/api`

Grader URL will be: `https://your-app.up.railway.app/api/status`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Physics engine | Python 3.11 — pure stdlib + numpy/scipy |
| Web framework | FastAPI 0.115 + uvicorn |
| Spatial index | scipy KDTree / custom 3D VoxelHash |
| Frontend | Vanilla JS + Canvas 2D API |
| Analytics | Streamlit + Altair + Pandas |
| Reverse proxy | Nginx |
| Container | Docker + Docker Compose |
| Base image | ubuntu:22.04 |

---

## 📊 Simulation Epoch

`2026-03-12T08:00:00Z` — all timestamps in ISO 8601 UTC.

---

*Orbital Insight — NSH 2026 · Built by **Team BroCODE** for IIT Delhi Hackathon*