# focal_loss.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared focal-loss objective for XGBoost training AND inference.
#
# WHY THIS FILE EXISTS:
#   joblib/pickle stores function references by module path.
#   When train_model.py runs as a script its __name__ == '__main__', so the
#   function would be pickled as  __main__.focal_loss_objective.
#   When uvicorn later loads main.py, uvicorn is __main__ — not your app —
#   so unpickling crashes with:
#       AttributeError: Can't get attribute 'focal_loss_objective'
#                       on <module '__main__' from '.../uvicorn'>
#
#   Keeping the function here gives it a stable, importable module path:
#       focal_loss.focal_loss_objective
#   which resolves correctly in every context.
#
# USAGE:
#   from focal_loss import focal_loss_objective, FOCAL_ALPHA, FOCAL_GAMMA
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

FOCAL_ALPHA: float = 0.25   # down-weight easy negatives
FOCAL_GAMMA: float = 2.0    # focus strength


def focal_loss_objective(y_pred: np.ndarray, dtrain) -> tuple:
    """
    XGBoost custom Focal Loss objective.

    Required by both train_model.py (training) and main.py (unpickling).
    Parameters must match exactly what was used during training.
    """
    y_true  = dtrain.get_label()
    p       = 1.0 / (1.0 + np.exp(-y_pred))
    p       = np.clip(p, 1e-7, 1.0 - 1e-7)
    alpha_t = np.where(y_true == 1, FOCAL_ALPHA, 1.0 - FOCAL_ALPHA)
    p_t     = np.where(y_true == 1, p, 1.0 - p)
    grad = alpha_t * (
        -FOCAL_GAMMA * (1.0 - p_t) ** (FOCAL_GAMMA - 1)
        * np.log(p_t + 1e-9) * p * (1.0 - p)
        + (1.0 - p_t) ** FOCAL_GAMMA * (p - y_true)
    )
    hess = alpha_t * (1.0 - p_t) ** FOCAL_GAMMA * p * (1.0 - p)
    hess = np.maximum(hess, 1e-6)
    return grad, hess
