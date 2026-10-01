"""
Open-set recognition: tells "a known appliance" apart from "something the
model has never seen".

The classifier always has to pick one of its known classes, so an unseen
appliance gets a confident wrong answer. This module adds a second opinion:
for every known appliance we store a few "typical" feature clusters (mean and
covariance). A new window is measured against them with the Mahalanobis
distance, which is a distance in units of "how many normal spreads away".
If the window is further out than that appliance's own limit, it is
reported as unknown.

Calibrated on this project's data (leave-one-class-out):
  - a held-out blender was flagged as unknown 100% of the time
  - a held-out iPad was NOT caught (it sits inside the laptop's range)
So this catches appliances that look electrically different, not look-alikes.

The fitted statistics are plain numpy arrays in a dict, saved by
train_model.py as openset.pkl.
"""
import numpy as np

# Indices into features.FEATURE_NAMES. We skip latest_V (mains voltage drifts
# through the day) and delta_W (noisy), keeping the 10 shape-of-the-load features.
COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K_COMPONENTS = 3      # typical clusters per appliance (an iron has "heating" and "cooling")
MARGIN = 3.0          # limit = MARGIN x the 99th percentile of the appliance's own training distances
MIN_COMPONENT = 15    # ignore clusters smaller than this
KMEANS_MIN_ROWS = 60  # fewer rows than this: use a single cluster
EXCLUDE = {"nothing_plugged", "warming_up", "unknown"}


def _components(A, k):
    if k > 1 and len(A) >= KMEANS_MIN_ROWS:
        from sklearn.cluster import KMeans
        scale = A.std(0) + 1e-6
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(A / scale)
    else:
        lab = np.zeros(len(A), dtype=int)
    comps = []
    for j in np.unique(lab):
        B = A[lab == j]
        if len(B) < MIN_COMPONENT:
            continue
        mu = B.mean(0)
        C = np.cov(B.T)
        sd2 = np.diag(C)
        # shrink toward the diagonal and add a small spread floor so a nearly
        # constant feature cannot make the distance explode
        C = 0.7 * C + 0.3 * np.diag(sd2) + np.diag((0.02 * np.maximum(np.abs(mu), 1.0)) ** 2)
        comps.append((mu, np.linalg.inv(C)))
    return comps


def _min_dist(comps, A):
    ds = []
    for mu, P in comps:
        Z = A - mu
        ds.append(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, P, Z), 0.0)))
    return np.min(np.stack(ds, 1), 1)


def fit(X, y, k=K_COMPONENTS, margin=MARGIN):
    """X: (n, 12) feature matrix from features.extract_features, y: labels."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = {}
    for c in sorted(set(y.tolist())):
        if c in EXCLUDE:
            continue
        A = X[y == c][:, COLS]
        comps = _components(A, k)
        if not comps:
            continue
        thr = float(np.quantile(_min_dist(comps, A), 0.99)) * margin
        classes[str(c)] = {"comps": comps, "thr": max(thr, 1e-6)}
    return {"version": 1, "cols": COLS, "margin": margin, "classes": classes}


def score(stats, features):
    """Returns (ratio, nearest_class). ratio > 1 means "outside every known appliance"."""
    if not stats or not stats.get("classes"):
        return 0.0, None
    a = np.asarray(features, dtype=float)[stats["cols"]][None, :]
    best, best_c = float("inf"), None
    for c, info in stats["classes"].items():
        r = float(_min_dist(info["comps"], a)[0]) / info["thr"]
        if r < best:
            best, best_c = r, c
    return best, best_c


def is_unknown(stats, features):
    ratio, nearest = score(stats, features)
    return ratio > 1.0, ratio, nearest
