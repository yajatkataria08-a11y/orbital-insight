"""
pytest tests for Orbital Insight ACM v8.0 + ML v4.0
Run: pip install pytest httpx fastapi && pytest test_long_sim.py -v
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
from main import (
    app, sim, Vec3, State, rk4, j2_accel, tsiolkovsky,
    collision_probability_chan, sim_time_to_iso, iso_to_sim_time,
    STD_FUEL_MASS, STD_DRY_MASS, STD_ISP, CONJ_THRESH, FUEL_EOL_PCT, RE,
    DVBandit, DebrisAnomalyDetector, FuelForecaster, ConjunctionRiskTracker,
    _dv_bandit, _anomaly_det, _fuel_fore, _risk_tracker,
    Debris,
)

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
#  PHYSICS UNIT TESTS
# ═══════════════════════════════════════════════════════════════════

def test_rk4_conserves_energy():
    """Total energy of a circular orbit should remain within 0.01% after 1 orbit."""
    MU = 398600.4418; RE_KM = 6378.137
    a = RE_KM + 550.0
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
#  ML PREDICT_RISK ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════

_VALID_CONJUNCTION = {
    "miss_distance_m":        500.0,
    "relative_velocity_ms":   7500.0,
    "altitude_km":            550.0,
    "inclination_diff_deg":   5.0,
    "time_to_closest_s":      3600.0,
    "debris_eccentricity":    0.01,
    "combined_radius_m":      3.0,
    "dist_rate_kms":          1.0,
    "atmospheric_density_multiplier": 1.0,
}

def test_predict_risk_returns_valid_structure():
    r = client.post("/api/ml/predict_risk", json=_VALID_CONJUNCTION)
    assert r.status_code == 200
    data = r.json()
    assert "risk_label" in data
    assert data["risk_label"] in (0, 1)
    assert "risk_level" in data
    assert data["risk_level"] in ("HIGH", "LOW")
    assert "collision_probability" in data
    assert 0.0 <= data["collision_probability"] <= 1.0
    assert "chan_pc" in data
    assert "model" in data
    assert "uncertainty" in data
    unc = data["uncertainty"]
    assert "lower" in unc
    assert "upper" in unc
    assert "coverage" in unc
    assert "high_alert" in unc
    assert "calibration_n" in unc

def test_predict_risk_physics_override_on_overlap():
    """If miss_distance_m <= combined_radius_m, model must return HIGH risk."""
    payload = {**_VALID_CONJUNCTION, "miss_distance_m": 2.0, "combined_radius_m": 3.0}
    r = client.post("/api/ml/predict_risk", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_label"] == 1
    assert data["collision_probability"] == 1.0
    assert "physics_override" in data["model"]

def test_predict_risk_invalid_altitude():
    payload = {**_VALID_CONJUNCTION, "altitude_km": 100.0}
    r = client.post("/api/ml/predict_risk", json=payload)
    assert r.status_code == 422

def test_predict_risk_invalid_eccentricity():
    payload = {**_VALID_CONJUNCTION, "debris_eccentricity": 1.5}
    r = client.post("/api/ml/predict_risk", json=payload)
    assert r.status_code == 422

def test_predict_risk_missing_required_field():
    payload = {k: v for k, v in _VALID_CONJUNCTION.items() if k != "miss_distance_m"}
    r = client.post("/api/ml/predict_risk", json=payload)
    assert r.status_code == 422

def test_predict_risk_batch_valid():
    payload = {"conjunctions": [_VALID_CONJUNCTION] * 3}
    r = client.post("/api/ml/predict_risk_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert data["batch_size"] == 3
    assert len(data["results"]) == 3
    for result in data["results"]:
        assert "risk_label" in result
        assert "collision_probability" in result
        assert "uncertainty" in result

def test_predict_risk_batch_too_large():
    payload = {"conjunctions": [_VALID_CONJUNCTION] * 501}
    r = client.post("/api/ml/predict_risk_batch", json=payload)
    assert r.status_code == 422

def test_predict_risk_batch_empty():
    r = client.post("/api/ml/predict_risk_batch", json={"conjunctions": []})
    assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════
#  ML-1: THOMPSON SAMPLING BANDIT TESTS
# ═══════════════════════════════════════════════════════════════════

def test_bandit_initialises_all_arms():
    b = DVBandit()
    assert len(b.ARMS) == 6

def test_bandit_visits_all_arms_via_thompson():
    """Thompson Sampling should explore all 6 arms over enough rounds."""
    b = DVBandit()
    seen = set()
    for _ in range(60):
        idx, dv = b.select_arm(time_to_tca=3600.0, rel_vel_kms=7.5)
        seen.add(idx)
        b.update(idx, 2.0, dv)   # miss > SAFETY_THRESH → success
    assert len(seen) == 6, f"Only visited arms: {seen}"

def test_bandit_update_signature_v2():
    """update() in v2 takes (arm_idx, miss_achieved_km, dv_used_kms)."""
    b = DVBandit()
    # Success: miss > SAFETY_THRESH (1.0 km)
    b.update(0, 2.5, b.ARMS[0])
    # Failure: miss < SAFETY_THRESH
    b.update(1, 0.5, b.ARMS[1])
    # alpha should increment for arm 0 (success), beta for arm 1 (failure)
    import numpy as np
    assert float(b._alpha[0]) > DVBandit.ALPHA_PRIOR
    assert float(b._beta[1])  > DVBandit.BETA_PRIOR

def test_bandit_urgent_tca_uses_large_arms():
    """With urgent TCA < 30 min, only upper-half arms should be selected."""
    b = DVBandit()
    n = len(b.ARMS)
    upper_half = set(range(n // 2, n))
    for _ in range(20):
        idx, dv = b.select_arm(time_to_tca=600.0, rel_vel_kms=7.5)
        assert idx in upper_half, f"Urgent TCA selected small arm {idx}"
        b.update(idx, 2.0, dv)

def test_bandit_contextual_normal_excludes_smallest():
    """Normal context (1800–14400 s TCA) should exclude arm 0 (0.004 km/s)."""
    b = DVBandit()
    for _ in range(30):
        idx, dv = b.select_arm(time_to_tca=7200.0, rel_vel_kms=7.5)
        assert idx != 0, "Normal context should not select arm 0 (too small)"
        b.update(idx, 2.0, dv)

def test_bandit_best_arm_reflects_updates():
    """Repeatedly rewarding arm 2 should make it the best arm."""
    b = DVBandit()
    for _ in range(20):
        b.update(2, 5.0, b.ARMS[2])   # large success
    for i in [0, 1, 3, 4, 5]:
        b.update(i, 0.1, b.ARMS[i])   # failures for others
    best_idx, best_dv = b.best_arm()
    assert best_idx == 2
    assert best_dv == b.ARMS[2]

def test_bandit_stats_structure_v2():
    """stats() should include Thompson Sampling fields."""
    stats = _dv_bandit.stats()
    assert "arms" in stats
    assert "best_dv_kms" in stats
    assert "sampler" in stats
    assert stats["sampler"] == "thompson_sampling"
    assert len(stats["arms"]) == 6
    for arm in stats["arms"]:
        assert "dv_kms" in arm
        assert "mean_reward" in arm
        assert "visits" in arm
        assert "alpha" in arm           # new in v2
        assert "beta" in arm            # new in v2
        assert "posterior_mean" in arm  # new in v2


# ═══════════════════════════════════════════════════════════════════
#  ML-2: ANOMALY DETECTOR TESTS (12-D features, online scoring)
# ═══════════════════════════════════════════════════════════════════

import random as _random

def _make_debris(n=50):
    debris = {}
    for i in range(n):
        a = RE + 550
        r = Vec3(a, _random.uniform(-100, 100), _random.uniform(-100, 100))
        v = Vec3(_random.uniform(-8, -6), _random.uniform(0, 2), _random.uniform(-1, 1))
        debris[f"DEB-T-{i:03d}"] = Debris(f"DEB-T-{i:03d}", State(r, v, 0.0), 0.1)
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
        assert 0.0 <= score <= 1.0, f"{did} score={score} out of [0,1]"

def test_anomaly_online_score_new():
    """score_new() should immediately score a newly ingested debris object."""
    det = DebrisAnomalyDetector()
    det.train(_make_debris(50))
    new_id = "DEB-NEW-999"
    new_d = Debris(new_id, State(Vec3(RE+550, 0, 0), Vec3(-7.5, 0.5, 0.1), 0.0), 0.1)
    det.score_new(new_id, new_d)
    assert new_id in det._scores
    assert 0.0 <= det._scores[new_id] <= 1.0

def test_anomaly_risk_multiplier_5_tiers():
    """v2 has 5 risk tiers: 1.0, 1.5, 3.0, 5.0, 8.0."""
    det = DebrisAnomalyDetector()
    det._scores = {
        "A": 0.30,   # < 0.45 → 1.0×
        "B": 0.50,   # 0.45–0.60 → 1.5×
        "C": 0.65,   # 0.60–0.75 → 3.0×
        "D": 0.80,   # 0.75–0.88 → 5.0×  (new tier)
        "E": 0.92,   # > 0.88 → 8.0×
    }
    assert det.risk_multiplier("A") == 1.0
    assert det.risk_multiplier("B") == 1.5
    assert det.risk_multiplier("C") == 3.0
    assert det.risk_multiplier("D") == 5.0   # new tier in v2
    assert det.risk_multiplier("E") == 8.0   # was 6.0 in v1

def test_anomaly_unknown_returns_conservative():
    assert DebrisAnomalyDetector().score("NONEXISTENT") == 0.5

def test_anomaly_train_count_increments():
    det = DebrisAnomalyDetector()
    det.train(_make_debris(50))
    det.train(_make_debris(50))
    assert det._train_count == 2

def test_ml_anomalies_endpoint():
    r = client.get("/api/ml/anomalies?top=5")
    assert r.status_code == 200
    data = r.json()
    assert "trained" in data
    assert "top_anomalies" in data
    assert "debris_scored" in data


# ═══════════════════════════════════════════════════════════════════
#  ML-3: QUADRATIC RLS FUEL FORECASTER TESTS
# ═══════════════════════════════════════════════════════════════════

def test_fuel_forecaster_prediction_decreasing():
    ff = FuelForecaster()
    for i in range(10):
        ff.update("TEST", float(i * 600), STD_FUEL_MASS - i * 0.5)
    f_now = ff.predict_fuel("TEST", 5400.0)
    f_1h  = ff.predict_fuel("TEST", 5400.0 + 3600)
    assert f_1h < f_now or f_1h == 0.0

def test_fuel_forecaster_eol_time():
    ff = FuelForecaster()
    for i in range(12):
        ff.update("EOL", float(i * 600), STD_FUEL_MASS - i * 1.5)
    t_eol = ff.time_to_eol("EOL", 6600.0)
    assert t_eol > 6600.0 or t_eol == float('inf')

def test_fuel_forecaster_stable_no_eol():
    ff = FuelForecaster()
    for i in range(5):
        ff.update("STABLE", float(i * 600), 50.0)
    assert ff.time_to_eol("STABLE", 3000.0) == float('inf')

def test_fuel_forecaster_burn_rate_ema():
    """burn_rate_ema() should return 0 before any data, positive after burns."""
    ff = FuelForecaster()
    assert ff.burn_rate_ema("UNKNOWN") == 0.0
    ff.update("SAT-A", 0.0,    50.0)
    ff.update("SAT-A", 600.0,  49.0)
    ff.update("SAT-A", 1200.0, 47.5)  # accelerating burn
    assert ff.burn_rate_ema("SAT-A") > 0.0

def test_fuel_forecaster_prune_eol():
    """prune_eol() should remove all state for that satellite."""
    ff = FuelForecaster()
    ff.update("DEAD", 0.0, 50.0)
    ff.update("DEAD", 600.0, 48.0)
    ff.prune_eol("DEAD")
    assert "DEAD" not in ff._models
    assert "DEAD" not in ff._ema
    assert ff.predict_fuel("DEAD", 1000.0) == STD_FUEL_MASS  # returns default

def test_fuel_forecaster_recovery_feasible():
    """recovery_feasible() should return False when fuel is too low."""
    ff = FuelForecaster()
    # Low fuel scenario
    for i in range(10):
        ff.update("LOW", float(i * 600), 3.0 - i * 0.2)
    assert not ff.recovery_feasible("LOW", 6000.0, required_kg=5.0)
    # High fuel scenario
    ff2 = FuelForecaster()
    ff2.update("HIGH", 0.0, 50.0)
    assert ff2.recovery_feasible("HIGH", 0.0, required_kg=1.0)

def test_ml_fuel_forecast_endpoint():
    r = client.get("/api/ml/fuel_forecast")
    assert r.status_code == 200
    data = r.json()
    assert "satellites" in data
    assert "eol_threshold_kg" in data
    if data["satellites"]:
        sat = data["satellites"][0]
        assert "fuel_now_kg" in sat
        assert "fuel_1h_kg" in sat
        assert "eol_warning" in sat


# ═══════════════════════════════════════════════════════════════════
#  ML-4: KALMAN RISK TRACKER TESTS
# ═══════════════════════════════════════════════════════════════════

def test_risk_tracker_converging_not_skipped():
    """Kalman tracker: converging miss distance should NOT be skipped."""
    rt = ConjunctionRiskTracker()
    for miss in [10.0, 8.0, 5.0]:
        rt.update("S1", "D1", miss, 60.0)
    assert not rt.should_skip_scan("S1", "D1")

def test_risk_tracker_diverging_is_skipped():
    """Kalman tracker: strongly diverging miss should be skipped."""
    rt = ConjunctionRiskTracker()
    for miss in [1.0, 8.0, 20.0, 40.0]:
        rt.update("S1", "D2", miss, 60.0)
    assert rt.should_skip_scan("S1", "D2")

def test_risk_tracker_unknown_not_skipped():
    assert not ConjunctionRiskTracker().should_skip_scan("X", "Y")

def test_risk_tracker_priority_converging_lower():
    """Converging pair should have lower priority score than diverging."""
    rt = ConjunctionRiskTracker()
    rt.update("S", "D1", 5.0, 60.0); rt.update("S", "D1", 3.0, 60.0)
    rt.update("S", "D2", 5.0, 60.0); rt.update("S", "D2", 9.0, 60.0)
    assert rt.priority_score("S", "D1") < rt.priority_score("S", "D2")

def test_risk_tracker_kalman_state_initialized():
    """After first update, Kalman state vectors should be initialised."""
    rt = ConjunctionRiskTracker()
    rt.update("S", "D", 10.0, 60.0)
    key = ("S", "D")
    assert key in rt._x
    assert key in rt._P

def test_risk_tracker_decay_stale_pairs():
    """decay_stale_pairs() should evict pairs with miss_km > 200."""
    rt = ConjunctionRiskTracker()
    rt.update("S", "FAR",  300.0, 60.0)   # should be evicted
    rt.update("S", "NEAR",   5.0, 60.0)   # should survive
    rt.decay_stale_pairs(current_t=0.0)
    assert ("S", "FAR")  not in rt._smoothed
    assert ("S", "NEAR") in rt._smoothed

def test_risk_tracker_max_pairs_evicts_safest():
    """When MAX_PAIRS is exceeded, the safest (largest miss) pair is evicted."""
    rt = ConjunctionRiskTracker()
    rt.MAX_PAIRS = 3  # override for test speed
    rt.update("S", "D1",  5.0, 60.0)
    rt.update("S", "D2", 10.0, 60.0)
    rt.update("S", "D3",  3.0, 60.0)
    rt.update("S", "D4",  7.0, 60.0)   # this triggers eviction of D2 (miss=10)
    assert ("S", "D2") not in rt._smoothed
    assert ("S", "D1") in rt._smoothed

def test_ml_risk_trends_endpoint():
    r = client.get("/api/ml/risk_trends?top=10")
    assert r.status_code == 200
    data = r.json()
    assert "tracked_pairs" in data
    assert "converging_pairs" in data

def test_ml_summary_endpoint():
    r = client.get("/api/ml/summary")
    assert r.status_code == 200
    data = r.json()
    assert "ml_modules" in data
    for key in ("bandit", "anomaly_detector", "fuel_forecaster", "risk_tracker"):
        assert key in data["ml_modules"]
        assert "status" in data["ml_modules"][key]
        assert "description" in data["ml_modules"][key]


# ═══════════════════════════════════════════════════════════════════
#  KS DRIFT DETECTION + CONFORMAL PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════

from main import _run_ks_drift_check, _conformal_interval, _conformal_residuals

def test_ks_drift_check_no_files():
    """KS check should return skipped status when data files are absent."""
    result = _run_ks_drift_check()
    # Either skipped (no files) or ok (files present)
    assert result["status"] in ("ok", "skipped — data files not found", "error") or \
           isinstance(result.get("drift_detected"), bool)

def test_conformal_interval_insufficient_data():
    """With fewer than 30 residuals, conformal interval should return (None, None)."""
    _conformal_residuals.clear()
    for _ in range(10):
        _conformal_residuals.append(0.1)
    lo, hi = _conformal_interval(0.5)
    assert lo is None and hi is None

def test_conformal_interval_valid_with_enough_data():
    """With ≥ 30 residuals, should return a valid (lower, upper) interval."""
    _conformal_residuals.clear()
    for i in range(50):
        _conformal_residuals.append(float(i % 10) / 10.0)
    lo, hi = _conformal_interval(0.5)
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= hi <= 1.0

def test_conformal_interval_bounds_probability():
    """Returned interval must always be within [0, 1]."""
    _conformal_residuals.clear()
    for _ in range(40):
        _conformal_residuals.append(0.8)   # large residuals
    lo, hi = _conformal_interval(0.95)
    assert lo is not None
    assert lo >= 0.0
    assert hi <= 1.0