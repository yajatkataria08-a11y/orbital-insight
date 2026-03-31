import numpy as np
import pandas as pd
import json
import math

# ── Physical constants — must match main.py exactly ───────────────────────────
MU                    = 398600.4418   # km³/s²
RE                    = 6378.137      # km
SAT_RADIUS            = 0.002         # km  (2 m)
DEB_RADIUS            = 0.001         # km  (1 m)
COMB_RADIUS           = SAT_RADIUS + DEB_RADIUS   # 0.003 km = 3 m
PC_MANEUVER_THRESHOLD = 1e-6

# ── Satellite shell inclinations from main.py _build() ────────────────────────
SAT_INCLINATIONS = [53.0, 70.0, 97.6]   # degrees

# ── Number of synthetic debris objects to simulate ────────────────────────────
# GroupKFold in train_model.py groups data by debris_id so no single debris
# object appears in both train and validation folds.  We simulate N_DEBRIS
# distinct objects; each is assigned a random number of encounters (1–20),
# giving a realistic long-tailed distribution where some active debris objects
# generate many conjunction events and most only generate a few.
N_DEBRIS = 3_000


def chan_pc(miss_km: float, rel_vel_kms: float,
            combined_r_km: float = COMB_RADIUS) -> float:
    """Exact replica of collision_probability_chan() in main.py."""
    sigma = max(0.05, miss_km * 0.3)
    A_cb  = math.pi * combined_r_km ** 2
    pc    = (A_cb / (2.0 * math.pi * sigma ** 2)) * math.exp(
                -miss_km ** 2 / (2.0 * sigma ** 2))
    return float(min(1.0, max(0.0, pc)))


def generate_training_data(n_samples: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Stratified sampling zones ──────────────────────────────────────────────
    #
    #   Zone A — DANGER (30 %): miss < 100 m, hypervelocity → almost always risk=1
    #   Zone B — HARD NEGATIVES (15 %): miss 500 m–2 km BUT Pc still < threshold
    #            because velocity is low or combined_radius is tiny.
    #            Teaches the model: close ≠ collision.
    #   Zone C — SAFE / BORDERLINE (55 %): wide miss range, full velocity spread.
    #
    n_danger   = int(n_samples * 0.30)
    n_hard_neg = int(n_samples * 0.15)
    n_safe     = n_samples - n_danger - n_hard_neg

    # ── Zone A: danger (miss < 100 m, hypervelocity) ──────────────────────────
    miss_a    = rng.uniform(0.001,  0.100,  n_danger)   # km
    vel_a_ms  = rng.uniform(3_000,  15_000, n_danger)
    alt_a     = rng.uniform(300,    800,    n_danger)
    tca_a     = rng.uniform(0,      7_200,  n_danger)
    ecc_a     = rng.uniform(0.0,    0.05,   n_danger)
    comb_a    = rng.uniform(3.0,    20.0,   n_danger)   # m — large debris fragments

    # ── Zone B: hard negatives (moderate miss, LOW velocity / tiny body) ───────
    miss_b    = rng.uniform(0.5,    2.0,    n_hard_neg)  # km
    vel_b_ms  = rng.uniform(50,     2_000,  n_hard_neg)  # m/s — slow approach
    alt_b     = rng.uniform(300,    800,    n_hard_neg)
    tca_b     = rng.uniform(0,      86_400, n_hard_neg)
    ecc_b     = rng.uniform(0.0,    0.05,   n_hard_neg)
    comb_b    = rng.uniform(0.5,    2.0,    n_hard_neg)  # m — CubeSat-class

    # ── Zone C: safe / borderline (wide range, learns decision boundary) ───────
    miss_c    = rng.uniform(0.05,   10.0,   n_safe)
    vel_c_ms  = rng.uniform(100,    15_000, n_safe)
    alt_c     = rng.uniform(300,    800,    n_safe)
    tca_c     = rng.uniform(0,      86_400, n_safe)
    ecc_c     = rng.uniform(0.0,    0.05,   n_safe)
    comb_c    = rng.uniform(0.5,    20.0,   n_safe)

    # ── Concatenate all zones ───────────────────────────────────────────────────
    miss_km  = np.concatenate([miss_a,   miss_b,   miss_c])
    vel_ms   = np.concatenate([vel_a_ms, vel_b_ms, vel_c_ms])
    alt_km   = np.concatenate([alt_a,    alt_b,    alt_c])
    tca_s    = np.concatenate([tca_a,    tca_b,    tca_c])
    ecc      = np.concatenate([ecc_a,    ecc_b,    ecc_c])
    comb_r_m = np.concatenate([comb_a,   comb_b,   comb_c])   # metres

    # ── inclination_diff_deg ───────────────────────────────────────────────────
    sat_inc  = rng.choice(SAT_INCLINATIONS, size=n_samples)
    deb_inc  = rng.uniform(0.0, 100.0, n_samples)
    inc_diff = np.abs(sat_inc - deb_inc)

    # ── dist_rate_kms ──────────────────────────────────────────────────────────
    approach_angle_deg = rng.uniform(0.0, 90.0, n_samples)
    approach_cos       = np.cos(np.radians(approach_angle_deg))
    vel_kms            = vel_ms / 1_000.0
    dist_rate_kms      = vel_kms * approach_cos

    # ── Gaussian measurement noise ─────────────────────────────────────────────
    def noisy(arr: np.ndarray, pct: float = 0.02) -> np.ndarray:
        return arr * (1.0 + rng.normal(0.0, pct, len(arr)))

    def noisy_abs(arr: np.ndarray, sigma_m: float) -> np.ndarray:
        return arr + rng.normal(0.0, sigma_m, len(arr))

    miss_km       = np.clip(noisy_abs(miss_km * 1_000.0, 5.0) / 1_000.0, 1e-4, 50.0)
    vel_ms        = np.clip(noisy(vel_ms),          50.0,   20_000.0)
    alt_km        = np.clip(noisy(alt_km),          150.0,  2_000.0)
    inc_diff      = np.clip(noisy(inc_diff),        0.0,    180.0)
    tca_s         = np.clip(noisy(tca_s),           0.0,    86_400.0)
    ecc           = np.clip(noisy(ecc),             0.0,    0.99)
    comb_r_m      = np.clip(noisy(comb_r_m),        0.5,    20.0)
    dist_rate_kms = np.clip(noisy(dist_rate_kms, 0.05), -20.0, 20.0)

    # ── Solar weather / atmospheric density multiplier ────────────────────────
    adm_category = rng.choice([0, 1, 2], size=n_samples, p=[0.70, 0.20, 0.10])
    adm_base = np.where(adm_category == 0,
                        rng.uniform(0.8, 1.2, n_samples),
               np.where(adm_category == 1,
                        rng.uniform(1.5, 3.0, n_samples),
                        rng.uniform(3.5, 6.0, n_samples)))
    atmospheric_density_multiplier = np.clip(adm_base, 0.5, 6.0)

    # ── v2.0: Engineered features ──────────────────────────────────────────────
    kinetic_energy_proxy = (vel_ms / 1_000.0) ** 2
    log_miss_distance_m  = np.log1p(miss_km * 1_000.0)

    # ── v3.0: Time-series delta features ──────────────────────────────────────
    SIM_DT_S     = 30.0
    miss_m_arr   = miss_km * 1_000.0
    dist_rate_ms = dist_rate_kms * 1_000.0
    miss_t1      = np.clip(miss_m_arr + dist_rate_ms * SIM_DT_S,       0.1, 1e6)
    miss_t2      = np.clip(miss_m_arr + dist_rate_ms * 2 * SIM_DT_S,   0.1, 1e6)
    delta_miss_m_per_s   = (miss_m_arr - miss_t1) / SIM_DT_S
    distance_acceleration = (miss_m_arr - 2 * miss_t1 + miss_t2) / (SIM_DT_S ** 2)

    # ── v3.0: Advanced physics features ───────────────────────────────────────
    r_km           = RE + alt_km
    grav_potential = -MU / r_km
    inc_diff_rad   = np.radians(inc_diff)
    sin_inc_diff   = np.sin(inc_diff_rad)
    cos_inc_diff   = np.cos(inc_diff_rad)

    # ── v4.0: RTN velocity components (Physics-Informed) ──────────────────────
    # Decompose scalar relative velocity into Radial, Transverse, Normal components.
    # vel_t_ms (transverse/along-track) is the most collision-critical direction.
    # Collisions with high vel_t_ms have shorter warning times (TCA approaches fast).
    #
    # Approximation using inclination difference:
    #   Normal component: out-of-plane → proportional to sin(inc_diff / 2)
    #   In-plane residual split into Radial (= closing rate) and Transverse (residual)
    inc_half_rad  = np.radians(inc_diff / 2.0)
    vel_n_ms      = np.minimum(np.abs(np.sin(inc_half_rad)) * vel_ms, vel_ms)
    vel_in_plane  = np.sqrt(np.maximum(0.0, vel_ms ** 2 - vel_n_ms ** 2))
    vel_r_ms      = dist_rate_ms                                           # radial = closing rate
    vel_t_ms      = np.sqrt(np.maximum(0.0, vel_in_plane ** 2 - vel_r_ms ** 2))  # transverse

    # ── v4.0: Log-probability feature (Chan Formula Prior) ────────────────────
    # Provides the physics-based Chan Pc as a log-scale feature.
    # The model uses this as a baseline and only "corrects" it when data shows
    # patterns the formula misses (e.g. high-eccentricity fast-movers).
    comb_r_km_arr = comb_r_m / 1_000.0
    sigma_arr     = np.maximum(0.05, miss_km * 0.3)
    A_cb_arr      = math.pi * comb_r_km_arr ** 2
    pc_arr_v4     = np.clip(
        (A_cb_arr / (2.0 * math.pi * sigma_arr ** 2))
        * np.exp(-miss_km ** 2 / (2.0 * sigma_arr ** 2)),
        0.0, 1.0
    )
    log_chan_pc   = np.log10(np.maximum(pc_arr_v4, 1e-15))  # log10(Pc), range [-15, 0]

    # ── v4.0: Orbital Period Ratio (Resonance) ─────────────────────────────────
    # Satellites and debris in resonant orbits (ratio ≈ 1.0 or 2.0) encounter
    # each other repeatedly, making them much more dangerous than the current-pass
    # miss distance alone suggests.
    #
    # For same-shell debris (most common case), ratio → 1.0 (highest risk).
    # We add realistic spread: debris eccentricity shifts the semi-major axis
    # slightly, creating small but meaningful period differences.
    a_sat_km     = RE + alt_km
    T_sat        = 2.0 * math.pi * np.sqrt(a_sat_km ** 3 / MU)
    # Debris semi-major axis: perturbed by eccentricity (a_deb = a_sat * (1 + ecc*0.1))
    a_deb_km     = a_sat_km * (1.0 + ecc * 0.1 * rng.uniform(-1.0, 1.0, n_samples))
    a_deb_km     = np.maximum(a_deb_km, RE + 200.0)
    T_deb        = 2.0 * math.pi * np.sqrt(a_deb_km ** 3 / MU)
    period_ratio = np.clip(T_sat / T_deb, 0.5, 2.0)

    # ── v4.0: debris_id for GroupKFold ────────────────────────────────────────
    # Assign each sample a debris_id from a pool of N_DEBRIS objects.
    # Each debris object generates a random number of encounter events (1–20),
    # matching the real-world distribution where active debris objects create
    # many repeated conjunction events and most only generate a few.
    #
    # GroupKFold in train_model.py groups by debris_id so the model is forced
    # to generalise to unseen debris objects rather than memorising specific
    # trajectories — critical for a deployable space safety system.
    debris_pool = [f"DEB-{i:05d}" for i in range(N_DEBRIS)]
    # Weight debris objects by a Zipf distribution (a few objects dominate)
    zipf_weights = 1.0 / (np.arange(1, N_DEBRIS + 1) ** 0.8)
    zipf_weights /= zipf_weights.sum()
    debris_ids   = rng.choice(debris_pool, size=n_samples, p=zipf_weights)

    # ── Physics-grounded labels via Chan Pc — VECTORISED ──────────────────────
    pc_arr  = np.clip(
        (A_cb_arr / (2.0 * math.pi * sigma_arr ** 2))
        * np.exp(-miss_km ** 2 / (2.0 * sigma_arr ** 2)),
        0.0, 1.0
    )
    labels  = (pc_arr > PC_MANEUVER_THRESHOLD).astype(int)

    # ── Build DataFrame ────────────────────────────────────────────────────────
    # Column ORDER must match REQUIRED_FEATURES in train_model.py v4.0 exactly.
    df = pd.DataFrame({
        # ── Primary features (v1.0 — original 8) ─────────────────────────────
        "miss_distance_m":                miss_km * 1_000.0,
        "relative_velocity_ms":           vel_ms,
        "altitude_km":                    alt_km,
        "inclination_diff_deg":           inc_diff,
        "time_to_closest_s":              tca_s,
        "debris_eccentricity":            ecc,
        "combined_radius_m":              comb_r_m,
        "dist_rate_kms":                  dist_rate_kms,
        # ── Engineered features (v2.0) ────────────────────────────────────────
        "kinetic_energy_proxy":           kinetic_energy_proxy,
        "log_miss_distance_m":            log_miss_distance_m,
        # ── Time-series delta features (v3.0) ─────────────────────────────────
        "delta_miss_m_per_s":             delta_miss_m_per_s,
        "distance_acceleration":          distance_acceleration,
        # ── Advanced engineered features (v3.0) ───────────────────────────────
        "grav_potential":                 grav_potential,
        "sin_inc_diff":                   sin_inc_diff,
        "cos_inc_diff":                   cos_inc_diff,
        "atmospheric_density_multiplier": atmospheric_density_multiplier,
        # ── NEW v4.0: Physics-Informed RTN Components ─────────────────────────
        "vel_r_ms":                       vel_r_ms,
        "vel_t_ms":                       vel_t_ms,
        "vel_n_ms":                       vel_n_ms,
        # ── NEW v4.0: Log-Probability Chan Prior ──────────────────────────────
        "log_chan_pc":                    log_chan_pc,
        # ── NEW v4.0: Orbital Period Ratio (Resonance) ────────────────────────
        "period_ratio":                   period_ratio,
        # ── GroupKFold key (NOT a model feature) ──────────────────────────────
        "debris_id":                      debris_ids,
        # ── Extra analysis columns (NOT fed to model) ─────────────────────────
        "miss_distance_km":               miss_km,
        "relative_velocity_kms":          vel_ms / 1_000.0,
        "chan_pc":                         pc_arr,
        # ── Label ─────────────────────────────────────────────────────────────
        "risk":                           labels,
    })

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv("training_data.csv", index=False)

    print(f"Generated {n_samples} samples "
          f"({n_danger} danger / {n_hard_neg} hard-neg / {n_safe} safe+border)")
    print(df["risk"].value_counts().to_string())
    print(f"\nClass balance:   {df['risk'].mean() * 100:.1f}% positive (risk=1)")
    print(f"Unique debris:   {df['debris_id'].nunique()} objects "
          f"(GroupKFold will prevent leakage)")

    # ── Feature list — MUST match REQUIRED_FEATURES in train_model.py v4.0 ────
    features = [
        # v1.0 — raw orbital parameters
        "miss_distance_m",
        "relative_velocity_ms",
        "altitude_km",
        "inclination_diff_deg",
        "time_to_closest_s",
        "debris_eccentricity",
        "combined_radius_m",
        "dist_rate_kms",
        # v2.0 — engineered
        "kinetic_energy_proxy",
        "log_miss_distance_m",
        # v3.0 — time-series delta
        "delta_miss_m_per_s",
        "distance_acceleration",
        # v3.0 — advanced orbital physics
        "grav_potential",
        "sin_inc_diff",
        "cos_inc_diff",
        "atmospheric_density_multiplier",
        # v4.0 — RTN velocity decomposition (physics-informed)
        "vel_r_ms",
        "vel_t_ms",
        "vel_n_ms",
        # v4.0 — Chan formula log-probability prior
        "log_chan_pc",
        # v4.0 — orbital resonance ratio
        "period_ratio",
    ]

    with open("feature_names.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"\nFeature list saved → feature_names.json ({len(features)} features):")
    for i, feat in enumerate(features, 1):
        version = "v1.0" if i <= 8 else "v2.0" if i <= 10 else "v3.0" if i <= 16 else "v4.0"
        print(f"  {i:2}. [{version}] {feat}")

    return df


if __name__ == "__main__":
    generate_training_data(n_samples=100_000)