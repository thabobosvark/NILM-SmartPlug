"""
Shared feature extraction - used by train_model.py AND webapp.py.
Keep this the single source of truth for how a window becomes a feature
vector, so training and live serving never drift apart.
"""

WINDOW_SIZE = 5  # ~35s of history at 7s polling

FEATURE_NAMES = [
    "latest_V", "latest_I", "latest_W", "latest_PF",
    "mean_W", "std_W", "min_W", "max_W", "range_W",
    "mean_PF", "std_PF",
    "delta_W",
]

# Instant-off / iron-sticky constants (used by webapp.py)
ZERO_W_THRESHOLD = 1.0
ZERO_I_THRESHOLD = 0.02
IRON_HOLD_SECONDS = 60


def _std(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def extract_features(window):
    if len(window) == 0:
        raise ValueError("window is empty")

    latest = window[-1]
    powers = [r["W"] for r in window]
    pfs = [r["PF"] for r in window]

    mean_W = sum(powers) / len(powers)
    max_W = max(powers)
    min_W = min(powers)

    delta_W = window[-1]["W"] - window[-2]["W"] if len(window) >= 2 else 0.0

    return [
        latest["V"], latest["I"], latest["W"], latest["PF"],
        mean_W, _std(powers), min_W, max_W, max_W - min_W,
        sum(pfs) / len(pfs), _std(pfs),
        delta_W,
    ]