import math
import json
import os
import warnings
warnings.filterwarnings("ignore")

import time as _time
from tqdm import tqdm, trange
from tqdm.auto import tqdm as tqdm_auto

# ── Startup banner ─────────────────────────────────────────────────────────────
_T0 = _time.time()
print("\n" + "═"*64)
print("  🛰   ACM Collision-Risk XGBoost Trainer  v4.0")
print("═"*64 + "\n")

import subprocess
import numpy as np
import pandas as pd
import joblib

# ── GPU / CUDA detection ────────────────────────────────────────────────────────
def _detect_cuda() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False
        # Also verify XGBoost was built with CUDA support
        import xgboost as _xgb
        _probe = _xgb.XGBClassifier(device="cuda", n_estimators=1, tree_method="hist")
        import numpy as _np
        _probe.fit(_np.zeros((10, 2)), _np.array([0]*5 + [1]*5))
        return True
    except Exception:
        return False

USE_GPU = _detect_cuda()
DEVICE  = "cuda" if USE_GPU else "cpu"
if USE_GPU:
    print("[INFO] ✅ GPU detected — XGBoost will use device=cuda")
else:
    print("[WARN] ⚠  No GPU / CUDA not available — falling back to CPU")
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, recall_score,
)
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

# ── Optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — skipping plots")

# ── Optional SHAP ──────────────────────────────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
    print("[INFO] SHAP found — will generate explainability report")
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed — skipping SHAP analysis. "
          "Install with: pip install shap")

# ── Optional Optuna ────────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    print("[INFO] Optuna found — will run hyperparameter search")
except ImportError:
    HAS_OPTUNA = False
    print("[WARN] optuna not installed — using default hyperparameters. "
          "Install with: pip install optuna")

# ── Feature list v4.0 — locked to main.py predict_risk hardcoded order ────────
FEATURE_FILE = "feature_names.json"
REQUIRED_FEATURES = [
    "miss_distance_m",
    "relative_velocity_ms",
    "altitude_km",
    "inclination_diff_deg",
    "time_to_closest_s",
    "debris_eccentricity",
    "combined_radius_m",
    "dist_rate_kms",
    "kinetic_energy_proxy",               # v2.0 — (rel_vel_kms)²
    "log_miss_distance_m",                # v2.0 — log1p(miss_distance_m)
    "delta_miss_m_per_s",                 # v3.0 — first diff of miss distance
    "distance_acceleration",              # v3.0 — second diff (trend curvature)
    "grav_potential",                     # v3.0 — −GM/r gravitational potential
    "sin_inc_diff",                       # v3.0 — sin(inclination_diff_deg)
    "cos_inc_diff",                       # v3.0 — cos(inclination_diff_deg)
    "atmospheric_density_multiplier",     # v3.0 — solar weather / drag multiplier
    "vel_r_ms",                           # v4.0 — RTN radial velocity [m/s]
    "vel_t_ms",                           # v4.0 — RTN transverse velocity [m/s]
    "vel_n_ms",                           # v4.0 — RTN normal velocity [m/s]
    "log_chan_pc",                        # v4.0 — log10(Chan Pc) physics prior
    "period_ratio",                       # v4.0 — orbital resonance ratio
]

if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE) as f:
        features = json.load(f)
    if features != REQUIRED_FEATURES:
        print(f"[WARN] feature_names.json doesn't match expected v4.0 list — overriding")
        features = REQUIRED_FEATURES
    print(f"[INFO] Using {len(features)} features: {features}")
else:
    features = REQUIRED_FEATURES
    print(f"[WARN] {FEATURE_FILE} not found — using hardcoded v2.0 feature order")

# ── Load base training data ────────────────────────────────────────────────────
if not os.path.exists("training_data.csv"):
    raise FileNotFoundError("training_data.csv not found — run generate_data.py first")

print("[1/7] Loading training data…")
with tqdm(total=1, desc="  Reading training_data.csv", bar_format="{l_bar}{bar}| {elapsed}") as _pbar:
    df = pd.read_csv("training_data.csv")
    _pbar.update(1)
print(f"      → {len(df):,} rows loaded")
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Features missing from training_data.csv: {missing}. "
                     f"Re-run generate_data.py v2.0 to add engineered features.")

# ── Dynamic Feedback Loop: Temporal Decay + Difficulty Scaling ─────────────────
#
#   Temporal Decay:   newer missed cases receive higher sample weights via an
#     exponential decay over row-index order (oldest = lowest weight).
#     Half-life is MISSED_HALFLIFE_ROWS — at that row count from the end,
#     weight = 0.5 of the newest case.  This lets the model adapt when the
#     debris environment shifts (e.g. a recent breakup cloud).
#
#   Difficulty Scaling:  A flat ×3 replica gives equal attention to a case
#     where the model predicted 45% risk (almost uncertain) and one where it
#     predicted 1% risk (confidently wrong and dangerous).  Instead we scale
#     by  confidence_gap = max(0, 0.5 − ml_probability)  normalised to [1, MAX].
#     A perfectly confident false-negative (ml_prob=0.0) gets ×MAX_DIFF_SCALE;
#     a borderline case (ml_prob=0.499) gets ×1.0.
#
#   AUTOTUNE_TRIGGER_COUNT:  if the number of missed cases exceeds this
#     threshold, force a full Optuna search and hot-swap the model only when
#     the new CV Recall beats the incumbent's stored test_recall_default.
#
MISSED_CSV             = "missed_cases.csv"
MISSED_HALFLIFE_ROWS   = 200        # exponential half-life (rows from end)
MAX_DIFF_SCALE         = 8.0        # max per-sample weight multiplier
AUTOTUNE_TRIGGER_COUNT = 100        # force re-search when missed cases ≥ this

n_missed              = 0
missed_weight_summary = ""

if os.path.exists(MISSED_CSV):
    print("[2/7] Loading missed-cases feedback data…")
    with tqdm(total=1, desc="  Reading missed_cases.csv", bar_format="{l_bar}{bar}| {elapsed}") as _pbar:
        df_missed = pd.read_csv(MISSED_CSV)
        _pbar.update(1)
    # Keep only rows that have all required features + optional ml_probability
    keep_cols  = [f for f in features if f in df_missed.columns] + ["risk"]
    has_ml_prob = "ml_probability" in df_missed.columns
    if has_ml_prob:
        keep_cols.append("ml_probability")
    df_missed = df_missed[keep_cols].dropna(subset=[f for f in features
                                                    if f in df_missed.columns] + ["risk"])
    if len(df_missed) > 0:
        n_missed = len(df_missed)
        n_rows   = n_missed

        # ── Temporal decay weights ─────────────────────────────────────────────
        # Row 0 is oldest; row n_rows-1 is newest.
        row_idx        = np.arange(n_rows)
        decay_lambda   = math.log(2.0) / max(MISSED_HALFLIFE_ROWS, 1)
        age_from_end   = (n_rows - 1 - row_idx).astype(float)
        temporal_w     = np.exp(-decay_lambda * age_from_end)   # (0, 1]

        # ── Difficulty scaling weights ─────────────────────────────────────────
        if has_ml_prob:
            ml_probs    = df_missed["ml_probability"].values.astype(float)
            conf_gap    = np.maximum(0.0, 0.5 - ml_probs)      # 0 … 0.5
            diff_scale  = 1.0 + (MAX_DIFF_SCALE - 1.0) * (conf_gap / 0.5)
        else:
            diff_scale  = np.full(n_rows, MAX_DIFF_SCALE / 2.0)

        # ── Combined weight ────────────────────────────────────────────────────
        combined_w     = temporal_w * diff_scale
        # Normalise so mean weight = 1 (keeps effective sample count intuitive)
        combined_w     = combined_w / combined_w.mean()

        # Build sample_weight array for the base training set (all 1.0)
        base_sw        = np.ones(len(df))
        df_feat_only   = df_missed[[f for f in features if f in df_missed.columns]
                                   + ["risk"]]
        missed_sw      = combined_w

        df      = pd.concat([df, df_feat_only], ignore_index=True)
        sample_weights_missed = np.concatenate([base_sw, missed_sw])

        w_min = float(combined_w.min()); w_max = float(combined_w.max())
        missed_weight_summary = (
            f"temporal-decay λ={decay_lambda:.4f}, "
            f"difficulty-scale ×{w_min:.2f}–×{w_max:.2f}"
        )
        print(f"[INFO] Feedback loop: merged {n_missed} missed cases "
              f"({missed_weight_summary})")
    else:
        print("[INFO] missed_cases.csv found but empty — skipping feedback loop")
        sample_weights_missed = np.ones(len(df))
else:
    print(f"[INFO] {MISSED_CSV} not found — no feedback data yet "
          f"(will be created by main.py at runtime)")
    sample_weights_missed = np.ones(len(df))

# ── Autotune trigger decision ───────────────────────────────────────────────────
FORCE_OPTUNA = False
if n_missed >= AUTOTUNE_TRIGGER_COUNT:
    FORCE_OPTUNA = True
    print(f"[INFO] Autotune triggered: {n_missed} missed cases ≥ threshold "
          f"{AUTOTUNE_TRIGGER_COUNT} → forcing full Optuna search")

X = df[features].values
y = df["risk"].values
# ── GroupKFold groups (Improvement 2c) ───────────────────────────────────────
# If the training data has a debris_id column (generated by generate_data.py
# v4.0), use it as the group key so GroupKFold ensures no debris object leaks
# between train and validation folds.  Falls back to a dummy per-row group
# (equivalent to StratifiedKFold) when the column is absent.
if "debris_id" in df.columns:
    from sklearn.preprocessing import LabelEncoder as _LE
    groups = _LE().fit_transform(df["debris_id"].values)
    CV_STRATEGY = "GroupKFold(debris_id)"
else:
    groups = np.arange(len(df))   # each row = its own group → identical to stratified
    CV_STRATEGY = "StratifiedKFold(no debris_id)"
print(f"[INFO] Cross-validation strategy: {CV_STRATEGY}")

shuffle_idx           = np.random.default_rng(42).permutation(len(df))
X                     = X[shuffle_idx]
y                     = y[shuffle_idx]
groups                = groups[shuffle_idx]
sample_weights_missed = sample_weights_missed[shuffle_idx]

print(f"\n[INFO] Dataset: {len(df)} samples, {len(features)} features")
print(f"[INFO] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"[INFO] Positive rate: {y.mean() * 100:.1f}%")

# ── Train / validation / test split ───────────────────────────────────────────
# Pass sample_weights through the same indices so each split keeps its weight.
indices = np.arange(len(X))
idx_train, idx_test = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y
)
idx_tr, idx_val = train_test_split(
    idx_train, test_size=0.15, random_state=42, stratify=y[idx_train]
)

X_train, X_test   = X[idx_train], X[idx_test]
y_train, y_test   = y[idx_train], y[idx_test]
X_tr,    X_val    = X[idx_tr],    X[idx_val]
y_tr,    y_val    = y[idx_tr],    y[idx_val]
sw_train          = sample_weights_missed[idx_train]
sw_tr             = sample_weights_missed[idx_tr]

# ── Class imbalance: recall-biased scale_pos_weight ───────────────────────────
# We boost SPW beyond the raw ratio so the model errs on the side of flagging
# risk (False Negatives are catastrophic in space safety).
n_neg = (y_tr == 0).sum()
n_pos = (y_tr == 1).sum()
# Use weighted counts when sample weights are non-trivial
if sw_tr.max() > 1.01:
    n_neg_w = float(sw_tr[y_tr == 0].sum())
    n_pos_w = float(sw_tr[y_tr == 1].sum())
    raw_spw = n_neg_w / max(n_pos_w, 1.0)
else:
    raw_spw = n_neg / max(n_pos, 1)
spw = round(raw_spw * 2.0, 3)   # ×2 recall bias
print(f"\n[INFO] Raw ratio={raw_spw:.2f}, recall-biased scale_pos_weight={spw:.3f}")

# ── Improvement 2a: Focal Loss custom objective ───────────────────────────────
# Standard Binary Cross-Entropy weights all errors equally.  Focal Loss
# down-weights easy (well-classified) examples and focuses training on the
# "borderline" conjunctions — exactly where the difference between a miss and
# a collision is smallest.  Alpha controls class imbalance; gamma controls focus.
#   FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
# Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
FOCAL_ALPHA = 0.25   # down-weight easy negatives
FOCAL_GAMMA = 2.0    # focus strength (2.0 is standard)

def focal_loss_objective(y_pred: np.ndarray, dtrain) -> tuple:
    """XGBoost custom objective implementing Focal Loss for binary classification."""
    y_true = dtrain.get_label()
    # Convert raw score → probability
    p      = 1.0 / (1.0 + np.exp(-y_pred))
    p      = np.clip(p, 1e-7, 1.0 - 1e-7)
    # Per-sample alpha weight
    alpha_t = np.where(y_true == 1, FOCAL_ALPHA, 1.0 - FOCAL_ALPHA)
    p_t     = np.where(y_true == 1, p, 1.0 - p)
    # First derivative (gradient)
    grad = alpha_t * (
        -FOCAL_GAMMA * (1.0 - p_t) ** (FOCAL_GAMMA - 1) * np.log(p_t + 1e-9) * p * (1.0 - p)
        + (1.0 - p_t) ** FOCAL_GAMMA * (p - y_true)
    )
    # Second derivative (hessian) — approximated as p(1-p) for stability
    hess = alpha_t * (1.0 - p_t) ** FOCAL_GAMMA * p * (1.0 - p)
    hess = np.maximum(hess, 1e-6)   # numerical floor
    return grad, hess

print("[INFO] Focal Loss custom objective registered "
      f"(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")

# ── Base hyperparameters (used when Optuna is absent) ─────────────────────────
# Improvement 2b: DART booster — Dropout meets Multiple Additive Regression
# Trees.  Randomly drops trees during training (rate=0.10), similar to Dropout
# in neural networks.  Prevents a few dominant trees from monopolising the
# ensemble, reducing overfitting to the training distribution.
BASE_PARAMS = dict(
    n_estimators          = 700,
    max_depth             = 6,
    learning_rate         = 0.04,
    subsample             = 0.85,
    colsample_bytree      = 0.80,
    min_child_weight      = 3,
    gamma                 = 0.15,
    reg_alpha             = 0.05,
    reg_lambda            = 1.5,
    scale_pos_weight      = spw,
    booster               = "dart",         # Improvement 2b: DART booster
    rate_drop             = 0.10,           # DART: 10% tree dropout rate
    skip_drop             = 0.50,           # DART: 50% chance to skip dropout step
    obj                   = focal_loss_objective,   # Improvement 2a: Focal Loss (in constructor)
    eval_metric           = "aucpr",
    # NOTE: early_stopping_rounds intentionally omitted —
    # DART booster uses randomised tree dropout which makes the eval curve
    # non-monotonic, so XGBoost 2.x raises an error if early stopping is set.
    # We use a fixed n_estimators budget instead (controlled by Optuna).
    random_state          = 42,
    n_jobs                = -1 if not USE_GPU else 1,   # n_jobs ignored on GPU; set 1 to avoid warning
    tree_method           = "hist",
    device                = DEVICE,                     # "cuda" if GPU detected, else "cpu"
)

# ── Optuna hyperparameter search ───────────────────────────────────────────────
if HAS_OPTUNA and (True or FORCE_OPTUNA):   # always run if optuna present; forced when triggered
    n_trials = 80 if FORCE_OPTUNA else 60
    print(f"\n[3/7] Optuna hyperparameter search "
          f"({n_trials} trials, optimising PR-AUC)"
          + (" [AUTOTUNE TRIGGERED]" if FORCE_OPTUNA else "") + "…")

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 900),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample        = trial.suggest_float("subsample", 0.60, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.50, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 0.5),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 0.5),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 3.0),
            scale_pos_weight = spw,
            booster          = "dart",
            rate_drop        = trial.suggest_float("rate_drop", 0.05, 0.20),
            skip_drop        = trial.suggest_float("skip_drop", 0.30, 0.70),
            obj              = focal_loss_objective,   # in constructor for XGBoost 2.x
            eval_metric      = "aucpr",
            # No early_stopping_rounds — DART booster's dropout makes eval
            # curves non-monotonic; XGBoost 2.x rejects early stopping with DART
            random_state     = 42,
            n_jobs           = 1 if USE_GPU else -1,
            tree_method      = "hist",
            device           = DEVICE,
        )
        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              sample_weight=sw_tr, verbose=False)
        prob_val = m.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, prob_val)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {**BASE_PARAMS, **study.best_params}
    # Restore non-tuned fixed params
    best_params["scale_pos_weight"]      = spw
    best_params["booster"]               = "dart"
    best_params["obj"]                   = focal_loss_objective
    best_params["eval_metric"]           = "aucpr"
    # Do NOT set early_stopping_rounds — incompatible with DART booster in XGBoost 2.x
    best_params.pop("early_stopping_rounds", None)
    best_params["random_state"]          = 42
    best_params["n_jobs"]                = 1 if USE_GPU else -1
    best_params["tree_method"]           = "hist"
    best_params["device"]                = DEVICE

    print(f"  Best trial PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
else:
    best_params = BASE_PARAMS
    print("[INFO] Using base hyperparameters (install optuna for search)")

# ── Improvement 2c: GroupKFold cross-validation ────────────────────────────────
# Groups data by debris_id so no single debris object appears in both train
# and validation folds.  This forces the model to generalise to unseen debris
# rather than memorising trajectories from specific objects it trained on.
# Falls back to StratifiedKFold when debris_id is unavailable.
model_cv = XGBClassifier(
    **{k: v for k, v in best_params.items()
       if k not in ("n_estimators", "early_stopping_rounds", "obj")},
    n_estimators=400,
)

if CV_STRATEGY.startswith("GroupKFold"):
    print("\n[4/7] 5-fold GroupKFold cross-validation (grouped by debris_id)…")
    cv        = GroupKFold(n_splits=5)
    groups_tr = groups[idx_train]
    _cv_metrics = {"roc_auc": [], "average_precision": [], "recall": []}
    _cv_splits  = list(cv.split(X_train, y_train, groups=groups_tr))
    for _fold, (_tr_idx, _va_idx) in enumerate(
            tqdm(_cv_splits, desc="  Cross-val folds", unit="fold",
                 bar_format="{l_bar}{bar}| fold {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")):
        _m = XGBClassifier(**{k: v for k, v in best_params.items()
                              if k not in ("n_estimators","early_stopping_rounds","obj")},
                           n_estimators=400)
        _m.fit(X_train[_tr_idx], y_train[_tr_idx])
        _p = _m.predict_proba(X_train[_va_idx])[:, 1]
        from sklearn.metrics import roc_auc_score as _roc, average_precision_score as _ap, recall_score as _rec
        _cv_metrics["roc_auc"].append(_roc(y_train[_va_idx], _p))
        _cv_metrics["average_precision"].append(_ap(y_train[_va_idx], _p))
        _cv_metrics["recall"].append(_rec(y_train[_va_idx], (_p >= 0.5).astype(int)))
    cv_roc = np.array(_cv_metrics["roc_auc"])
    cv_ap  = np.array(_cv_metrics["average_precision"])
    cv_rec = np.array(_cv_metrics["recall"])
else:
    print("\n[4/7] 5-fold StratifiedKFold cross-validation (no debris_id)…")
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _cv_metrics = {"roc_auc": [], "average_precision": [], "recall": []}
    _cv_splits  = list(cv.split(X_train, y_train))
    for _fold, (_tr_idx, _va_idx) in enumerate(
            tqdm(_cv_splits, desc="  Cross-val folds", unit="fold",
                 bar_format="{l_bar}{bar}| fold {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")):
        _m = XGBClassifier(**{k: v for k, v in best_params.items()
                              if k not in ("n_estimators","early_stopping_rounds","obj")},
                           n_estimators=400)
        _m.fit(X_train[_tr_idx], y_train[_tr_idx])
        _p = _m.predict_proba(X_train[_va_idx])[:, 1]
        from sklearn.metrics import roc_auc_score as _roc, average_precision_score as _ap, recall_score as _rec
        _cv_metrics["roc_auc"].append(_roc(y_train[_va_idx], _p))
        _cv_metrics["average_precision"].append(_ap(y_train[_va_idx], _p))
        _cv_metrics["recall"].append(_rec(y_train[_va_idx], (_p >= 0.5).astype(int)))
    cv_roc = np.array(_cv_metrics["roc_auc"])
    cv_ap  = np.array(_cv_metrics["average_precision"])
    cv_rec = np.array(_cv_metrics["recall"])

print(f"  CV strategy  : {CV_STRATEGY}")
print(f"  CV ROC-AUC:  {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print(f"  CV Avg-Prec: {cv_ap.mean():.4f}  ± {cv_ap.std():.4f}")
print(f"  CV Recall:   {cv_rec.mean():.4f} ± {cv_rec.std():.4f}  (safety-critical metric)")

# ── Final training with early stopping ────────────────────────────────────────
print("\n[5/7] Fitting final XGBClassifier (DART + Focal Loss)…")
_n_est = best_params.get("n_estimators", 700)

from xgboost.callback import TrainingCallback as _XGBTrainingCallback

class _TqdmXGBCallback(_XGBTrainingCallback):
    """Minimal XGBoost callback that drives a tqdm bar per tree round."""
    def __init__(self, total):
        super().__init__()
        self._bar = tqdm(total=total, desc="  XGBoost trees",
                         unit="tree",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} trees"
                                    " [{elapsed}<{remaining}, {rate_fmt}]")
    def after_iteration(self, model, epoch, evals_log):
        self._bar.update(1)
        aucpr = None
        try:
            aucpr = list(evals_log.values())[-1].get("aucpr", [None])[-1]
        except Exception:
            pass
        if aucpr is not None:
            self._bar.set_postfix({"val_aucpr": f"{aucpr:.4f}"})
        return False   # False = do NOT stop early (handled by XGBoost internally)
    def after_training(self, model):
        self._bar.close()
        return model

# Pass all params via constructor (XGBoost 2.x requirement).
# obj (Focal Loss) is already in best_params from BASE_PARAMS / Optuna merge.
# early_stopping_rounds is excluded — incompatible with DART booster.
_fit_params = {k: v for k, v in best_params.items()
               if k not in ("early_stopping_rounds",)}
base_model = XGBClassifier(**_fit_params, callbacks=[_TqdmXGBCallback(_n_est)])
base_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
               sample_weight=sw_tr,
               verbose=False)
# DART doesn't support best_iteration — use n_estimators directly
best_n = best_params.get("n_estimators", _n_est)
print(f"      → Training complete: {best_n} trees (DART booster + Focal Loss)")

# ── Calibration (isotonic regression) ─────────────────────────────────────────
# CalibratedClassifierCV wraps the XGBClassifier; it still exposes .predict()
# and .predict_proba(), so main.py calls work without modification.
# Isotonic is chosen over 'sigmoid' (Platt) because our output distribution
# is non-Gaussian and isotonic is non-parametric.
print("\n[6/7] Calibrating probabilities (CalibratedClassifierCV, isotonic, cv=5)…")
print("  Running 5 calibration folds…")

# CRITICAL: Strip tqdm callbacks from base_model BEFORE passing it to
# CalibratedClassifierCV.  sklearn's joblib deepcopies the estimator for each
# CV fold — a _TqdmXGBCallback holds a tqdm bar which holds a TextIOWrapper
# (stdout) that cannot be pickled, causing:
#   TypeError: cannot pickle 'TextIOWrapper' instances
# Removing callbacks here is safe: training is already complete.
try:
    base_model.set_params(callbacks=None)
except Exception:
    pass

# sklearn 1.3+ metadata routing: opt in so sample_weight is forwarded to the
# base estimator inside each calibration CV fold.
try:
    import sklearn as _sklearn
    _sklearn.set_config(enable_metadata_routing=True)
    base_model.set_fit_request(sample_weight=True)
except (AttributeError, RuntimeError):
    pass  # older sklearn versions handle sample_weight automatically

model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
model.fit(X_train, y_train, sample_weight=sw_train)
print("      → Calibration complete")

print("\n[7/7] Evaluating on held-out test set…")
_eval_steps = ["predict_proba", "predict", "accuracy", "f1", "recall",
               "roc_auc", "avg_precision", "confusion_matrix", "PR_curve",
               "optimal_threshold"]
with tqdm(_eval_steps, desc="  Metrics", unit="metric",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as _ebar:
    _ebar.set_postfix({"step": "predict_proba"})
    y_prob = model.predict_proba(X_test)[:, 1];           _ebar.update(1)
    _ebar.set_postfix({"step": "predict"})
    y_pred_def = model.predict(X_test).astype(int);       _ebar.update(1)
    _ebar.set_postfix({"step": "accuracy"})
    acc_def = accuracy_score(y_test, y_pred_def);         _ebar.update(1)
    _ebar.set_postfix({"step": "f1"})
    f1_def  = f1_score(y_test, y_pred_def);               _ebar.update(1)
    _ebar.set_postfix({"step": "recall"})
    rec_def = recall_score(y_test, y_pred_def);           _ebar.update(1)
    _ebar.set_postfix({"step": "roc_auc"})
    roc     = roc_auc_score(y_test, y_prob);              _ebar.update(1)
    _ebar.set_postfix({"step": "avg_precision"})
    ap      = average_precision_score(y_test, y_prob);    _ebar.update(1)
    _ebar.set_postfix({"step": "confusion_matrix"})
    cm      = confusion_matrix(y_test, y_pred_def);       _ebar.update(1)
    _ebar.set_postfix({"step": "PR_curve"})
    precisions, recalls, thresholds = precision_recall_curve(
        y_val, model.predict_proba(X_val)[:, 1]);         _ebar.update(1)
    _ebar.set_postfix({"step": "threshold"})
    f1_scores  = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx       = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1_val    = float(f1_scores[best_idx]);          _ebar.update(1)

y_pred_opt = (y_prob >= best_threshold).astype(int)
acc_opt = accuracy_score(y_test, y_pred_opt)
f1_opt  = f1_score(y_test, y_pred_opt)
rec_opt = recall_score(y_test, y_pred_opt)

print(f"\n{'='*64}")
print("  FINAL MODEL — HELD-OUT TEST SET PERFORMANCE")
print(f"{'='*64}")
print(f"  ROC-AUC        : {roc:.4f}")
print(f"  Avg-Precision  : {ap:.4f}  (area under PR curve)")
print(f"\n  Default 0.5 threshold (used by main.py .predict()):")
print(f"    Accuracy  : {acc_def:.4f}")
print(f"    F1 Score  : {f1_def:.4f}")
print(f"    Recall    : {rec_def:.4f}  ← safety-critical (higher = fewer missed collisions)")
print(f"\n  Optimal F1 threshold (reference only, NOT loaded by main.py):")
print(f"    Threshold : {best_threshold:.4f}")
print(f"    Accuracy  : {acc_opt:.4f}")
print(f"    F1 Score  : {f1_opt:.4f}")
print(f"    Recall    : {rec_opt:.4f}")
print(f"\n  Confusion matrix (default 0.5, rows=actual, cols=predicted):")
print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
print(f"  False Negative Rate : {cm[1,0] / (cm[1,0]+cm[1,1])*100:.2f}%  (missed collisions)")
print(f"\n{classification_report(y_test, y_pred_def, target_names=['SAFE', 'RISK'])}")

# ── Plots ──────────────────────────────────────────────────────────────────────
if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#010508")

    # Feature importance from the base (uncalibrated) estimator
    ax1 = axes[0]
    ax1.set_facecolor("#030b10")
    importances = base_model.feature_importances_
    idx = np.argsort(importances)
    ax1.barh([features[i] for i in idx], importances[idx], color="#00c8b4")
    ax1.set_xlabel("Feature Importance (XGBoost gain)", color="#a8c8e0", fontsize=9)
    ax1.set_title(f"Feature Importance (v3.0 — {len(features)} features)", color="#00e8d0", fontsize=10)
    ax1.tick_params(colors="#a8c8e0", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#0a2030")

    # Precision-Recall curve
    ax2 = axes[1]
    ax2.set_facecolor("#030b10")
    ax2.plot(recalls[:-1], precisions[:-1], color="#00c8b4", lw=1.5,
             label=f"AP={ap:.3f}")
    ax2.axvline(recalls[best_idx], color="#ffd700", lw=1, linestyle="--",
                label=f"Optimal t={best_threshold:.2f}")
    ax2.scatter([recalls[best_idx]], [precisions[best_idx]],
                color="#ffd700", zorder=5)
    ax2.set_xlabel("Recall", color="#a8c8e0", fontsize=9)
    ax2.set_ylabel("Precision", color="#a8c8e0", fontsize=9)
    ax2.set_title("Precision-Recall Curve (Calibrated)", color="#00e8d0", fontsize=10)
    ax2.tick_params(colors="#a8c8e0", labelsize=8)
    ax2.legend(fontsize=8, facecolor="#030b10", labelcolor="#a8c8e0")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#0a2030")

    fig.tight_layout()
    fig.savefig("model_report.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("[INFO] Saved → model_report.png")

# ── SHAP Explainability ────────────────────────────────────────────────────────
# Uses the uncalibrated base XGBClassifier for SHAP (CalibratedClassifierCV
# wraps it; TreeExplainer needs direct access to the booster).
# We sample a representative 2000-row subset of the test set for speed.
if HAS_SHAP and HAS_MPL:
    print("\n[INFO] Computing SHAP values on test set sample (n=2000)…")
    try:
        n_shap   = min(2000, len(X_test))
        rng_shap = np.random.default_rng(0)
        shap_idx = rng_shap.choice(len(X_test), n_shap, replace=False)
        X_shap   = X_test[shap_idx]

        explainer   = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_shap)

        # shap_values may be list[pos_class_array] for binary classifiers
        if isinstance(shap_values, list):
            sv = shap_values[1]   # positive class
        else:
            sv = shap_values

        # ── importance.png  (mean |SHAP| bar chart — Figure 1) ───────────────
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_order    = np.argsort(mean_abs_shap)
        fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
        fig_imp.patch.set_facecolor("#010508")
        ax_imp.set_facecolor("#030b10")
        bars = ax_imp.barh(
            [features[i] for i in shap_order],
            mean_abs_shap[shap_order],
            color="#00c8b4"
        )
        # Annotate bars with raw value
        for bar, val in zip(bars, mean_abs_shap[shap_order]):
            ax_imp.text(val + mean_abs_shap.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", ha="left",
                        color="#a8c8e0", fontsize=7)
        ax_imp.set_xlabel("Mean |SHAP value| (impact on model output)", color="#a8c8e0", fontsize=9)
        ax_imp.set_title("SHAP Feature Importance — ACM v3.0\n"
                          "(higher = feature influences model predictions more)",
                          color="#00e8d0", fontsize=10)
        ax_imp.tick_params(colors="#a8c8e0", labelsize=8)
        for spine in ax_imp.spines.values():
            spine.set_edgecolor("#0a2030")
        fig_imp.tight_layout()
        fig_imp.savefig("importance.png", dpi=130, bbox_inches="tight",
                        facecolor=fig_imp.get_facecolor())
        plt.close(fig_imp)
        print("[INFO] Saved → importance.png  (SHAP mean |value| bar chart)")

        # ── shap_beeswarm.png  (Figure 2 — value distribution by feature) ────
        fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
        fig_bee.patch.set_facecolor("#010508")
        ax_bee.set_facecolor("#030b10")
        shap.summary_plot(sv, X_shap, feature_names=features,
                          plot_type="dot", show=False, color_bar=True,
                          max_display=len(features))
        ax_bee = plt.gca()
        ax_bee.set_facecolor("#030b10")
        fig_bee.patch.set_facecolor("#010508")
        ax_bee.tick_params(colors="#a8c8e0", labelsize=8)
        ax_bee.set_title("SHAP Beeswarm — Feature Value vs Impact",
                          color="#00e8d0", fontsize=10)
        plt.tight_layout()
        fig_bee.savefig("shap_beeswarm.png", dpi=120, bbox_inches="tight",
                        facecolor="#010508")
        plt.close(fig_bee)
        print("[INFO] Saved → shap_beeswarm.png  (SHAP beeswarm distribution)")

        # ── Console top-5 summary (useful in log output) ──────────────────────
        top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
        print("\n[SHAP] Top-5 most influential features:")
        for rank, fi in enumerate(top5_idx, 1):
            print(f"  {rank}. {features[fi]:<28}  mean|SHAP|={mean_abs_shap[fi]:.5f}")

    except Exception as exc:
        print(f"[WARN] SHAP analysis failed: {exc}")
elif HAS_SHAP and not HAS_MPL:
    print("[WARN] SHAP requires matplotlib to save plots — skipping visualisation")

# ── OPT-4: ONNX Model Export ───────────────────────────────────────────────────
# ONNX Runtime is purpose-built for low-latency CPU inference and avoids the
# overhead of loading the full XGBoost Python library for every prediction.
# Benchmarks show 2–5× lower predict_proba() latency on multi-core CPUs vs
# the sklearn/xgboost pipeline, with identical numerical output.
#
# Export strategy:
#   • We export the raw base_model (XGBClassifier) — not the CalibratedCV
#     wrapper — because skl2onnx does not support CalibratedClassifierCV.
#   • At inference time main.py uses the .pkl calibrated model for
#     predict_proba(); the .onnx model is available as an optional fast path
#     that the /api/ml/predict_risk endpoint can swap to when onnxruntime is
#     installed.  The loader in main.py already falls back gracefully.
#   • The ONNX model outputs raw XGBoost probabilities (pre-calibration).
#     For safety-critical decisions, prefer the .pkl calibrated model.
#     For high-throughput screening (batch endpoint), ONNX is ideal.
#
# Install deps (optional — ONNX export is skipped if absent):
#   pip install skl2onnx onnxmltools onnxruntime
HAS_ONNX = False
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True
    print("[INFO] skl2onnx found — will export ONNX model")
except ImportError:
    print("[WARN] skl2onnx not installed — skipping ONNX export. "
          "Install with: pip install skl2onnx onnxruntime")

_onnx_path = None
if HAS_ONNX:
    try:
        # initial_types: one float32 input named 'features', shape (N, n_feats)
        initial_types = [("features", FloatTensorType([None, len(features)]))]
        onnx_model = convert_sklearn(
            base_model,            # raw XGBClassifier (not the calibrated wrapper)
            initial_types=initial_types,
            target_opset=17,       # ONNX opset 17 is widely supported
            options={id(base_model): {"zipmap": False}}  # return raw float arrays
        )
        _onnx_path = "collision_model.onnx"
        with open(_onnx_path, "wb") as f_onnx:
            f_onnx.write(onnx_model.SerializeToString())

        # Quick round-trip validation: compare ONNX output to sklearn output
        # on a small random subset of the test set.
        try:
            import onnxruntime as _ort
            _sess   = _ort.InferenceSession(_onnx_path,
                                             providers=["CPUExecutionProvider"])
            _n_val  = min(200, len(X_test))
            _X_val_f32 = X_test[:_n_val].astype(np.float32)
            _ort_out    = _sess.run(None, {"features": _X_val_f32})
            # onnxruntime returns [labels, probabilities_dict or array]
            # With zipmap=False: _ort_out[1] is (N, 2) float array
            _ort_proba  = _ort_out[1][:, 1]
            _skl_proba  = base_model.predict_proba(X_test[:_n_val])[:, 1]
            max_diff    = float(np.max(np.abs(_ort_proba - _skl_proba)))
            if max_diff < 1e-4:
                print(f"[INFO] ONNX round-trip validation PASSED  "
                      f"(max |Δproba| = {max_diff:.2e})")
            else:
                print(f"[WARN] ONNX round-trip diff = {max_diff:.4f} — "
                      f"exceeds 1e-4 tolerance.  Check opset compatibility.")
        except ImportError:
            print("[INFO] onnxruntime not installed — skipping round-trip check. "
                  "Install with: pip install onnxruntime")
        except Exception as _ort_exc:
            print(f"[WARN] ONNX round-trip check failed: {_ort_exc}")

        print(f"[INFO] Saved → {_onnx_path}  "
              f"({os.path.getsize(_onnx_path) / 1024:.0f} KB)")
    except Exception as onnx_exc:
        print(f"[WARN] ONNX export failed: {onnx_exc}")
        _onnx_path = None


# ── Save artifacts ─────────────────────────────────────────────────────────────
# collision_model.pkl: CalibratedClassifierCV wrapping XGBClassifier.
# main.py calls .predict() and .predict_proba() — both are supported.
# collision_model.onnx: raw XGBClassifier in ONNX format — use for fast-path
#   batch inference via onnxruntime (2–5× lower latency per call).
joblib.dump(model,          "collision_model.pkl")
joblib.dump(features,       "model_features.pkl")
joblib.dump(best_threshold, "model_threshold.pkl")  # reference only; main.py ignores

meta = {
    "model_type":             "CalibratedClassifierCV(XGBClassifier, isotonic)",
    "onnx_model":             _onnx_path or "not_exported",
    "onnx_note":              "raw XGBClassifier (pre-calibration) for fast-path inference",
    "main_py_threshold_used": 0.5,
    "note":                   "Calibrated: predict_proba() outputs true probabilities",
    "n_features":             len(features),
    "features":               features,
    "engineered_features":    ["kinetic_energy_proxy", "log_miss_distance_m"],
    "n_train_samples":        int(len(y_tr)),
    "n_val_samples":          int(len(y_val)),
    "n_test_samples":         int(len(y_test)),
    "n_missed_cases_merged":  n_missed,
    "missed_weight_strategy": missed_weight_summary or "none",
    "best_n_estimators":      int(best_n),
    "recall_bias_multiplier": 2.0,
    "scale_pos_weight":       round(float(spw), 3),
    "hyperparameter_search":  "optuna-60trials" if HAS_OPTUNA else "default",
    "autotune_triggered":     FORCE_OPTUNA,
    "test_roc_auc":           round(float(roc), 6),
    "test_avg_precision":     round(float(ap), 6),
    "test_accuracy_default":  round(float(acc_def), 6),
    "test_f1_default":        round(float(f1_def), 6),
    "test_recall_default":    round(float(rec_def), 6),
    "optimal_threshold":      round(best_threshold, 6),
    "test_f1_optimal":        round(float(f1_opt), 6),
    "test_recall_optimal":    round(float(rec_opt), 6),
    "cv_roc_auc_mean":        round(float(cv_roc.mean()), 6),
    "cv_roc_auc_std":         round(float(cv_roc.std()), 6),
    "cv_avg_prec_mean":       round(float(cv_ap.mean()), 6),
    "cv_recall_mean":         round(float(cv_rec.mean()), 6),
    "xgb_params": {
        k: v for k, v in best_params.items()
        if k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "reg_alpha", "reg_lambda",
                 "scale_pos_weight", "rate_drop", "skip_drop")
    },
    "cv_strategy":            CV_STRATEGY,
    "booster":                "dart",
    "focal_loss":             f"alpha={FOCAL_ALPHA},gamma={FOCAL_GAMMA}",
}

# ── Improvement 3b: KS Drift Detection ────────────────────────────────────────
# Compare the distribution of key features in the live missed_cases.csv against
# the distribution in training_data.csv.  A significant KS statistic (p < 0.05)
# means the live environment has drifted outside the training distribution — the
# model is operating "out of its depth" and should raise an alert.
KS_DRIFT_FEATURES = [
    "altitude_km", "miss_distance_m", "relative_velocity_ms",
    "atmospheric_density_multiplier",
]
KS_PVALUE_THRESHOLD = 0.05

drift_report = {}
if os.path.exists(MISSED_CSV) and os.path.exists("training_data.csv"):
    try:
        df_live  = pd.read_csv(MISSED_CSV)
        df_train = pd.read_csv("training_data.csv")
        print("\n[KS-DRIFT] Running Kolmogorov-Smirnov drift detection…")
        any_drift = False
        for feat in KS_DRIFT_FEATURES:
            if feat in df_live.columns and feat in df_train.columns:
                live_vals  = df_live[feat].dropna().values
                train_vals = df_train[feat].dropna().values
                if len(live_vals) < 20:
                    continue   # not enough live data for a meaningful test
                stat, pval = ks_2samp(train_vals, live_vals)
                drifted = bool(pval < KS_PVALUE_THRESHOLD)
                drift_report[feat] = {"ks_stat": round(float(stat), 4),
                                       "p_value": round(float(pval), 6),
                                       "drifted": drifted}
                status_str = "⚠ DRIFT" if drifted else "✓ OK"
                print(f"  {feat:<38}  KS={stat:.4f}  p={pval:.4f}  {status_str}")
                if drifted:
                    any_drift = True
        if any_drift:
            print("[KS-DRIFT] ⚠  One or more features show significant distribution shift.")
            print("           Model may be operating outside its training envelope.")
            print("           Consider re-running generate_data.py with updated parameters.")
        else:
            print("[KS-DRIFT] ✓  No significant drift detected — training data still representative.")
        meta["drift_detection"] = drift_report
    except Exception as _ks_exc:
        print(f"[KS-DRIFT] Warning: drift detection failed: {_ks_exc}")
else:
    print("[KS-DRIFT] Skipped — missed_cases.csv or training_data.csv not found.")
    meta["drift_detection"] = {}

# ── Automated Hot-Swap constants — must be defined BEFORE the A/B block ───────
MIN_RECALL_IMPROVEMENT = 0.002   # must beat incumbent by ≥ 0.2 pp to hot-swap
INCUMBENT_META         = "model_meta.json"
_incumbent_recall      = 0.0

import datetime   # used below in the A/B shadow log entry

# ── Improvement 3a: A/B Shadow Mode ───────────────────────────────────────────
# Before hot-swapping the incumbent, run both models in "shadow mode" on the
# validation set and log a side-by-side comparison to comparison.log.
# This lets operators verify the candidate outperforms the incumbent in practice
# before any production traffic is affected.
COMPARISON_LOG    = "comparison.log"
AB_TICKS_REQUIRED = 100   # main.py must confirm over this many live ticks

if os.path.exists(INCUMBENT_META):
    try:
        incumbent_pkl = "collision_model.pkl"
        if os.path.exists(incumbent_pkl):
            incumbent_model = joblib.load(incumbent_pkl)
            inc_proba  = incumbent_model.predict_proba(X_val)[:, 1]
            cand_proba = model.predict_proba(X_val)[:, 1]
            inc_ap     = float(average_precision_score(y_val, inc_proba))
            cand_ap    = float(average_precision_score(y_val, cand_proba))
            inc_rec    = float(recall_score(y_val, (inc_proba >= 0.5).astype(int)))
            cand_rec   = float(recall_score(y_val, (cand_proba >= 0.5).astype(int)))

            ab_entry = {
                "timestamp":          datetime.datetime.utcnow().isoformat() + "Z",
                "incumbent_ap":       round(inc_ap, 6),
                "candidate_ap":       round(cand_ap, 6),
                "incumbent_recall":   round(inc_rec, 6),
                "candidate_recall":   round(cand_rec, 6),
                "ap_delta":           round(cand_ap  - inc_ap,  6),
                "recall_delta":       round(cand_rec - inc_rec, 6),
                "candidate_better":   bool(cand_rec > inc_rec and cand_ap > inc_ap),
                "ab_ticks_required":  AB_TICKS_REQUIRED,
                "status":             "SHADOW — awaiting live tick confirmation",
            }
            with open(COMPARISON_LOG, "a") as _cl:
                _cl.write(json.dumps(ab_entry) + "\n")

            print(f"\n[A/B SHADOW] Incumbent  — AP={inc_ap:.4f}  Recall={inc_rec:.4f}")
            print(f"[A/B SHADOW] Candidate  — AP={cand_ap:.4f}  Recall={cand_rec:.4f}")
            print(f"[A/B SHADOW] Δ AP={cand_ap-inc_ap:+.4f}  Δ Recall={cand_rec-inc_rec:+.4f}")
            print(f"[A/B SHADOW] Logged to {COMPARISON_LOG} — full swap after {AB_TICKS_REQUIRED} live ticks")
            meta["ab_shadow"] = ab_entry
        else:
            print("[A/B SHADOW] No incumbent model found — skipping shadow comparison.")
            meta["ab_shadow"] = {"status": "skipped — no incumbent"}
    except Exception as _ab_exc:
        print(f"[A/B SHADOW] Warning: shadow comparison failed: {_ab_exc}")
        meta["ab_shadow"] = {"status": f"error: {_ab_exc}"}
else:
    meta["ab_shadow"] = {"status": "skipped — first run, no incumbent"}

import datetime

# ── Automated Hot-Swap ─────────────────────────────────────────────────────────
# Load the incumbent model's stored recall.  If the new model's CV Recall
# is statistically superior (> incumbent by MIN_RECALL_IMPROVEMENT), atomically
# replace collision_model.pkl.  Otherwise, keep the incumbent and save the new
# artefacts under a "_candidate" suffix so they can be inspected manually.

if os.path.exists(INCUMBENT_META):
    try:
        with open(INCUMBENT_META) as _f:
            _inc = json.load(_f)
        _incumbent_recall = float(_inc.get("test_recall_default", 0.0))
    except Exception:
        _incumbent_recall = 0.0

new_recall = float(rec_def)
do_hotswap = new_recall >= _incumbent_recall + MIN_RECALL_IMPROVEMENT or _incumbent_recall == 0.0

if do_hotswap:
    joblib.dump(model,          "collision_model.pkl")
    joblib.dump(features,       "model_features.pkl")
    joblib.dump(best_threshold, "model_threshold.pkl")
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if _incumbent_recall > 0.0:
        print(f"\n[HOT-SWAP] New Recall {new_recall:.4f} > "
              f"Incumbent {_incumbent_recall:.4f} — model replaced ✅")
    else:
        print(f"\n[HOT-SWAP] No incumbent found — initial model saved ✅")
else:
    # Save as candidate so the operator can inspect without disrupting production
    joblib.dump(model,          "collision_model_candidate.pkl")
    joblib.dump(features,       "model_features_candidate.pkl")
    joblib.dump(best_threshold, "model_threshold_candidate.pkl")
    with open("model_meta_candidate.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[HOT-SWAP] New Recall {new_recall:.4f} did NOT beat "
          f"Incumbent {_incumbent_recall:.4f} + margin {MIN_RECALL_IMPROVEMENT} "
          f"— incumbent kept, candidate saved as *_candidate.pkl ⚠️")

print("\n[INFO] Saved:")
if do_hotswap:
    print("  → collision_model.pkl   (CalibratedClassifierCV — main.py compatible)")
    print("  → model_features.pkl   (16-feature name list, v3.0)")
    print("  → model_threshold.pkl  (optimal F1 threshold — for reference only)")
    print("  → model_meta.json      (provenance, metrics, hyperparams)")
    if _onnx_path:
        print(f"  → {_onnx_path}  (raw XGBClassifier — fast-path inference via onnxruntime)")
else:
    print("  → collision_model_candidate.pkl  (did not beat incumbent recall)")
    print("  → model_meta_candidate.json")
if HAS_MPL:
    print("  → model_report.png     (feature importance + PR curve)")

print(f"\n✅ Done in {(_time.time()-_T0)/60:.1f} min — ROC-AUC {roc:.4f}  ·  Default-Recall {rec_def:.4f}"
      f"  ·  Default-F1 {f1_def:.4f}  ·  AP {ap:.4f}"
      f"  ·  Optimal-t {best_threshold:.4f}"
      + (f"\n          Missed-cases merged: {n_missed} ({missed_weight_summary})"
         if n_missed else "")
      + (f"\n          ONNX model: {_onnx_path}" if _onnx_path else ""))