"""
Edge anomaly model: top-K feature selection + one Mahalanobis model per class.

  python edge_model.py            # K = 4
  python edge_model.py --k 5

1. Ranks the 12 features with ExtraTrees importances.
2. Picks the top K, skipping latest_V / delta_W and any feature that is
   >0.95 correlated with one already chosen (keeps the covariance invertible).
3. Fits mean + inverse covariance per class on those K features, and a limit
   per class = 99th percentile of that class's own training distances.
Output: edge_stats.pkl

Also holds helpers shared by calibrate.py, autoencoder.py, drift.py, export_edge.py.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from features import FEATURE_NAMES
from train_model import assign_sessions, build_windowed_dataset

K_DEFAULT = 4
AVOID = {"latest_V", "delta_W"}          # mains voltage drifts; delta_W is noisy
EXCLUDE_CLASSES = {"nothing_plugged", "warming_up", "unknown"}
MIN_W = 1.0                              # same "a load is on" gate the ESP32 uses
CORR_MAX = 0.95
PCT = 0.99
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
POWER_COLS = [IDX[n] for n in ("latest_I", "latest_W", "mean_W", "std_W", "min_W", "max_W", "range_W")]


def load_data(csv_path="training_data.csv"):
    """Windowed features of known appliances only, load-on windows only."""
    df = assign_sessions(pd.read_csv(csv_path))
    X, y, g = build_windowed_dataset(df)
    keep = ~np.isin(y, list(EXCLUDE_CLASSES)) & (X[:, IDX["latest_W"]] >= MIN_W)
    return X[keep], y[keep], g[keep]


def split_masks(y, g, val_frac=0.25, gap=4):
    """Train/validation masks. Classes with >=2 sessions: whole sessions are held out.
    Single-session classes: the last 25% of the run is held out, with a gap of
    `gap` windows dropped between (windows overlap by 4 readings)."""
    tr = np.ones(len(y), bool)
    va = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        sess = sorted(set(g[idx].tolist()))
        if len(sess) >= 2:
            n_val = max(1, int(round(len(sess) * val_frac)))
            hit = idx[np.isin(g[idx], sess[-n_val:])]
            va[hit] = True
            tr[hit] = False
        else:
            n_val = int(len(idx) * val_frac)
            if n_val < 1:
                continue
            va[idx[-n_val:]] = True
            tr[idx[-(n_val + gap):]] = False
    return tr, va


def rank_features(X, y, k=K_DEFAULT):
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(pd.DataFrame(X, columns=FEATURE_NAMES), y)
    imp = clf.feature_importances_
    with np.errstate(all="ignore"):
        corr = np.nan_to_num(np.corrcoef(X.T))
    order = [i for i in np.argsort(-imp) if FEATURE_NAMES[i] not in AVOID]
    chosen = []
    for i in order:
        if all(abs(corr[i, j]) < CORR_MAX for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    for i in order:                      # fallback if the correlation filter was too strict
        if len(chosen) == k:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(int(i) for i in chosen), imp


def fit(X, y, cols, pct=PCT):
    classes = {}
    for c in sorted(set(y.tolist())):
        A = X[y == c][:, cols]
        if len(A) < 20:
            continue
        mu = A.mean(0)
        sd = A.std(0) + 1e-9
        C = np.cov(A.T)
        # shrink toward the diagonal + a spread floor so a near-constant feature can't explode the distance
        C = 0.8 * C + 0.2 * np.diag(np.diag(C)) + np.diag((0.02 * np.maximum(np.abs(mu), 1.0)) ** 2)
        P = np.linalg.inv(C)
        Z = A - mu
        d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, P, Z), 0.0))
        classes[str(c)] = {"mean": mu, "std": sd, "inv": P, "thr": float(np.quantile(d, pct)), "n": int(len(A))}
    return {"version": 2, "cols": [int(c) for c in cols],
            "names": [FEATURE_NAMES[i] for i in cols], "classes": classes}


def ratios(stats, X, claimed=None):
    """distance / class limit. claimed=None -> nearest class (label-free, what the ESP32 does).
    claimed=[labels] -> distance to that class only (server, using the classifier's label).
    Returns (ratio, class_index, class_names). ratio > 1 means 'beyond the normal limit'."""
    names = list(stats["classes"])
    A = np.asarray(X, float)[:, stats["cols"]]
    R = np.empty((len(A), len(names)))
    for j, c in enumerate(names):
        s = stats["classes"][c]
        Z = A - s["mean"]
        R[:, j] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, s["inv"], Z), 0.0)) / s["thr"]
    if claimed is None:
        return R.min(1), R.argmin(1), names
    col = np.array([names.index(c) for c in claimed])
    return R[np.arange(len(A)), col], col, names


def scale_power(F, factor):
    """Scale current and all power features together (a physically consistent surge/sag)."""
    G = np.array(F, float, copy=True)
    G[:, POWER_COLS] *= factor
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_data.csv")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    a = ap.parse_args()

    X, y, g = load_data(a.csv)
    print(f"{len(X)} load-on windows, {len(set(y))} appliance classes")
    cols, imp = rank_features(X, y, a.k)

    print("\nFeature importances (* = selected):")
    for i in np.argsort(-imp):
        print(f"  {'*' if i in cols else ' '} {FEATURE_NAMES[i]:10s} {imp[i]:.3f}")

    stats = fit(X, y, cols)
    joblib.dump(stats, "edge_stats.pkl")

    print("\nPer-class model:")
    for c, s in stats["classes"].items():
        print(f"  {c:12s} n={s['n']:5d}  limit(99th pct distance)={s['thr']:.2f}")

    r, _, _ = ratios(stats, X)
    print(f"\nSelf-check: {np.mean(r > 1.0):.2%} of training windows are above the limit (expect ~1%)")

    spl = pd.Series(g).groupby(pd.Series(y)).nunique()
    thin = spl[spl < 2].index.tolist()
    if thin:
        print(f"WARNING: only 1 session for {thin}; calibration for these uses a time-block hold-out, so it is optimistic.")
    print("\nSaved edge_stats.pkl")


if __name__ == "__main__":
    main()