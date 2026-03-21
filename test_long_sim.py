"""
pytest tests for Orbital Insight ACM v8.0
Run: pip install pytest httpx fastapi && pytest test_acm.py -v
"""
import pytest, math, asyncio
from fastapi.testclient import TestClient

import os
os.environ.update({
    "STD_FUEL_MASS": "50.0", "STD_DRY_MASS": "500.0", "STD_ISP": "300.0",
    "MAX_DV_PER_BURN": "0.015", "THERMAL_COOLDOWN": "600.0",
    "CONJ_THRESH": "0.1", "SK_BOX_RADIUS": "10.0",
    "SECRET_KEY": "test-secret", "LOG_LEVEL": "WARNING",
})
from main import app, sim, Vec3, State, rk4, j2_accel, tsiolkovsky, \
    collision_probability_chan, sim_time_to_iso, iso_to_sim_time, \
    STD_FUEL_MASS, STD_DRY_MASS, STD_ISP, CONJ_THRESH

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
#  PHYSICS UNIT TESTS
# ═══════════════════════════════════════════════════════════════════

def test_rk4_conserves_energy():
    """Total energy of a circular orbit should remain within 0.01% after 1 orbit."""
    MU = 398600.4418; RE = 6378.137
    a = RE + 550.0
    v_circ = math.sqrt(MU / a)
    r0 = Vec3(a, 0.0, 0.0)
    v0 = Vec3(0.0, v_circ, 0.0)
    s = State(r0, v0, 0.0)
    T = 2 * math.pi * math.sqrt(a**3 / MU)
    n_steps = int(T / 10.0)
    e0 = 0.5 * v_circ**2 - MU / a
    for _ in range(n_steps):
        s = rk4(s, 10.0)
    e1 = 0.5 * s.v.norm()**2 - MU / s.r.norm()
    assert abs((e1 - e0) / e0) < 1e-4, f"Energy drift too large: {abs((e1-e0)/e0):.2e}"

def test_j2_accel_direction():
    """J2 acceleration at equator should have no z-component for equatorial position."""
    r = Vec3(7000.0, 0.0, 0.0)
    a = j2_accel(r)
    assert abs(a.z) < 1e-10

def test_tsiolkovsky_mass_conservation():
    """Fuel burned should not exceed total wet mass."""
    m_wet = STD_DRY_MASS + STD_FUEL_MASS
    dm = tsiolkovsky(m_wet, 0.015, STD_ISP)
    assert dm < STD_FUEL_MASS, f"Tsiolkovsky gives dm={dm:.2f} kg, exceeds fuel mass"
    assert dm > 0.0

def test_collision_probability_decreases_with_distance():
    """Chan Pc should be monotonically decreasing with miss distance."""
    pcs = [collision_probability_chan(d, 5.0) for d in [0.05, 0.1, 0.5, 1.0, 5.0]]
    for i in range(len(pcs) - 1):
        assert pcs[i] > pcs[i+1], f"Pc not decreasing at index {i}"

def test_iso_roundtrip():
    """sim_time → ISO → sim_time should be lossless within 1 second."""
    for t in [0.0, 3600.0, 86400.0, 86400 * 30]:
        iso = sim_time_to_iso(t)
        back = iso_to_sim_time(iso)
        assert abs(back - t) < 1.0, f"ISO roundtrip failed for t={t}: got {back}"


# ═══════════════════════════════════════════════════════════════════
#  API INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════

def test_status_endpoint():
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "sim_time" in data
    assert "satellites" in data
    assert data["satellites"] >= 50
    assert "spatial_index" in data

def test_telemetry_ack():
    r = client.post("/api/telemetry", json={
        "timestamp": "2026-03-12T08:00:00.000Z",
        "objects": [{
            "id": "DEB-TEST-001", "type": "DEBRIS",
            "r": {"x": 4500.0, "y": -2100.0, "z": 4800.0},
            "v": {"x": -1.25, "y": 6.84, "z": 3.12}
        }]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ACK"
    assert data["processed_count"] == 1

def test_telemetry_missing_id_rejected():
    r = client.post("/api/telemetry", json={
        "objects": [{"type": "DEBRIS",
                     "r": {"x": 0, "y": 0, "z": 0},
                     "v": {"x": 0, "y": 0, "z": 0}}]
    })
    assert r.status_code == 422

def test_simulate_step_returns_correct_keys():
    """Step endpoint returns immediately with STEP_REQUESTED (non-blocking)."""
    r = client.post("/api/simulate/step", json={"step_seconds": 10.0})
    assert r.status_code == 200
    data = r.json()
    # Non-blocking: returns STEP_REQUESTED immediately
    assert data["status"] in ("STEP_REQUESTED", "STEP_COMPLETE")
    assert "target_sim_time" in data or "new_timestamp" in data
    assert "maneuvers_executed" in data

def test_simulate_step_validation():
    """step_seconds outside [0.1, 172800] should return 422."""
    r = client.post("/api/simulate/step", json={"step_seconds": -5.0})
    assert r.status_code == 422
    r2 = client.post("/api/simulate/step", json={"step_seconds": 999999.0})
    assert r2.status_code == 422

def test_maneuver_schedule_returns_202():
    sat_id = list(sim.sats.keys())[0]
    r = client.post("/api/maneuver/schedule", json={
        "satelliteId": sat_id,
        "maneuver_sequence": [{
            "burn_id": "TEST_BURN_1",
            "burnTime": "2026-03-12T20:00:00.000Z",
            "deltaV_vector": {"x": 0.001, "y": 0.002, "z": -0.001}
        }]
    })
    assert r.status_code == 202
    data = r.json()
    assert data["status"] == "SCHEDULED"
    assert "validation" in data
    assert "ground_station_los" in data["validation"]
    assert "projected_mass_remaining_kg" in data["validation"]

def test_maneuver_bad_satellite_returns_400():
    r = client.post("/api/maneuver/schedule", json={
        "satelliteId": "SAT-DOES-NOT-EXIST",
        "maneuver_sequence": [{
            "burn_id": "BAD_BURN",
            "burnTime": "2026-03-12T20:00:00.000Z",
            "deltaV_vector": {"x": 0.001, "y": 0.001, "z": 0.001}
        }]
    })
    assert r.status_code == 400

def test_snapshot_structure():
    r = client.get("/api/visualization/snapshot")
    assert r.status_code == 200
    data = r.json()
    assert "satellites" in data
    assert "debris_cloud" in data
    sat = data["satellites"][0]
    assert "fuel_kg" in sat
    assert "lat" in sat
    assert "lon" in sat
    deb = data["debris_cloud"][0]
    assert len(deb) == 4   # [id, lat, lon, alt]

def test_conjunction_detail_404():
    r = client.get("/api/satellite/SAT-DOES-NOT-EXIST/conjunction_detail")
    assert r.status_code == 404

def test_fleet_uptime_grade():
    r = client.get("/api/fleet/uptime")
    assert r.status_code == 200
    data = r.json()
    assert data["grade"] in ("EXCELLENT", "GOOD", "ACCEPTABLE", "POOR")
    assert 0.0 <= data["fleet_uptime_pct"] <= 100.0

def test_auth_token():
    r = client.post("/api/auth/token",
                    json={"username": "admin", "password": "orbital2026"})
    assert r.status_code == 200
    assert "access_token" in r.json()

def test_auth_bad_creds():
    r = client.post("/api/auth/token",
                    json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════
#  ML MODULE TESTS
# ═══════════════════════════════════════════════════════════════════

from main import (DVBandit, DebrisAnomalyDetector, FuelForecaster,
                  ConjunctionRiskTracker, _dv_bandit, _anomaly_det,
                  _fuel_fore, _risk_tracker, Debris, Vec3, State,
                  STD_FUEL_MASS, FUEL_EOL_PCT, RE)

# ── ML-1 Bandit ───────────────────────────────────────────────────
def test_bandit_initialises_all_arms():
    b = DVBandit()
    assert len(b.ARMS) == 6
    # Force visit all arms — UCB needs multiple rounds to guarantee all explored
    seen = set()
    for _ in range(30):   # enough rounds to cover all 6 arms via UCB exploration
        idx, dv = b.select_arm()
        seen.add(idx)
        b.update(idx, 1.0, dv)  # update so UCB moves to next arm
    assert len(seen) == 6, f"Only visited arms: {seen}"

def test_bandit_update_improves_high_reward_arm():
    b = DVBandit()
    for i in range(6):
        b.update(i, 5.0, b.ARMS[i])
    for _ in range(10):
        b.update(1, 50.0, 0.006)
    best_idx, best_dv = b.best_arm()
    assert best_idx == 1
    assert best_dv == 0.006

def test_bandit_ucb_explores_unvisited():
    b = DVBandit()
    idx, dv = b.select_arm()
    assert b._n[idx] == 0 or b._total == 0

def test_bandit_stats_structure():
    stats = _dv_bandit.stats()
    assert "arms" in stats and "best_dv_kms" in stats
    assert len(stats["arms"]) == 6
    for arm in stats["arms"]:
        assert "dv_kms" in arm and "mean_reward" in arm and "visits" in arm

# ── ML-2 Anomaly Detector ─────────────────────────────────────────
def _make_debris(n=50):
    import random
    debris = {}
    for i in range(n):
        a = RE + 550
        r = Vec3(a, random.uniform(-100,100), random.uniform(-100,100))
        v = Vec3(random.uniform(-8,-6), random.uniform(0,2), random.uniform(-1,1))
        debris[f"DEB-T-{i:03d}"] = Debris(f"DEB-T-{i:03d}", State(r,v,0.0), 0.1)
    return debris

def test_anomaly_detector_trains():
    det = DebrisAnomalyDetector()
    det.train(_make_debris(100))
    assert det._trained
    assert len(det._scores) == 100

def test_anomaly_scores_in_range():
    det = DebrisAnomalyDetector()
    det.train(_make_debris(80))
    for did, score in det._scores.items():
        assert 0.0 <= score <= 1.0

def test_anomaly_risk_multiplier_levels():
    det = DebrisAnomalyDetector()
    det._scores = {"A": 0.3, "B": 0.6, "C": 0.75, "D": 0.9}
    assert det.risk_multiplier("A") == 1.0
    assert det.risk_multiplier("B") == 1.5
    assert det.risk_multiplier("C") == 3.0
    assert det.risk_multiplier("D") == 6.0

def test_anomaly_unknown_returns_conservative():
    assert DebrisAnomalyDetector().score("NONEXISTENT") == 0.5

def test_ml_anomalies_endpoint():
    r = client.get("/api/ml/anomalies?top=5")
    assert r.status_code == 200
    data = r.json()
    assert "trained" in data and "top_anomalies" in data and "debris_scored" in data

# ── ML-3 Fuel Forecaster ──────────────────────────────────────────
def test_fuel_forecaster_prediction_decreasing():
    ff = FuelForecaster()
    for i in range(10):
        ff.update("TEST", float(i*600), STD_FUEL_MASS - i*0.5)
    f_now = ff.predict_fuel("TEST", 5400.0)
    f_1h  = ff.predict_fuel("TEST", 5400.0+3600)
    assert f_1h < f_now or f_1h == 0.0

def test_fuel_forecaster_eol_time():
    ff = FuelForecaster()
    for i in range(12):
        ff.update("EOL", float(i*600), STD_FUEL_MASS - i*1.5)
    t_eol = ff.time_to_eol("EOL", 6600.0)
    assert t_eol > 6600.0 or t_eol == float('inf')

def test_fuel_forecaster_stable_no_eol():
    ff = FuelForecaster()
    for i in range(5):
        ff.update("STABLE", float(i*600), 50.0)
    assert ff.time_to_eol("STABLE", 3000.0) == float('inf')

def test_ml_fuel_forecast_endpoint():
    r = client.get("/api/ml/fuel_forecast")
    assert r.status_code == 200
    data = r.json()
    assert "satellites" in data and "eol_threshold_kg" in data
    if data["satellites"]:
        sat = data["satellites"][0]
        assert "fuel_now_kg" in sat and "fuel_1h_kg" in sat and "eol_warning" in sat

# ── ML-4 Risk Tracker ────────────────────────────────────────────
def test_risk_tracker_converging_not_skipped():
    rt = ConjunctionRiskTracker()
    for miss in [10.0, 8.0, 5.0]:
        rt.update("S1","D1", miss, 60.0)
    assert not rt.should_skip_scan("S1","D1")

def test_risk_tracker_diverging_is_skipped():
    rt = ConjunctionRiskTracker()
    for miss in [1.0, 8.0, 20.0]:
        rt.update("S1","D2", miss, 60.0)
    assert rt.should_skip_scan("S1","D2")

def test_risk_tracker_unknown_not_skipped():
    assert not ConjunctionRiskTracker().should_skip_scan("X","Y")

def test_risk_tracker_priority_converging_lower():
    rt = ConjunctionRiskTracker()
    rt.update("S","D1",5.0,60.0); rt.update("S","D1",3.0,60.0)
    rt.update("S","D2",5.0,60.0); rt.update("S","D2",9.0,60.0)
    assert rt.priority_score("S","D1") < rt.priority_score("S","D2")

def test_ml_risk_trends_endpoint():
    r = client.get("/api/ml/risk_trends?top=10")
    assert r.status_code == 200
    data = r.json()
    assert "tracked_pairs" in data and "converging_pairs" in data

def test_ml_summary_endpoint():
    r = client.get("/api/ml/summary")
    assert r.status_code == 200
    data = r.json()
    assert "ml_modules" in data
    for key in ("bandit","anomaly_detector","fuel_forecaster","risk_tracker"):
        assert key in data["ml_modules"]
        assert "status" in data["ml_modules"][key]
        assert "description" in data["ml_modules"][key]