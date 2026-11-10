"""
Drift detection: EWMA of each class's feature means, compared with the
training baseline. Catches slow change (sensor gain drift, an appliance
ageing) that single-window anomaly checks miss.

  python drift.py --demo                  # verify on your data
  python drift.py --demo --limit 1.5

Drift score = max over features of |EWMA - training mean| / training std.
The EWMA starts AT the baseline, so a healthy appliance sits near 0.

Server use (same alert dict format as anomaly.py):
    mon = DriftMonitor(joblib.load("edge_stats.pkl"))
    alerts = mon.update(ts, label, conf, features12)   # features12 = features.extract_features(window)
"""
import argparse
import json

import joblib
import numpy as np

import edge_model as em

ALPHA = 0.02          # ~50-window memory (about 6 min at 7 s polling)
LIMIT = 2.0           # run the demo to tune this
MIN_CONF = 0.7
COOLDOWN_S = 3600


class DriftMonitor:
    def __init__(self, stats, alpha=ALPHA, limit=LIMIT):
        self.stats, self.alpha, self.limit = stats, alpha, limit
        self.cols = stats["cols"]
        self.ewma = {c: s["mean"].copy() for c, s in stats["classes"].items()}
        self.n = {c: 0 for c in stats["classes"]}
        self._last = {}

    def score(self, label):
        s = self.stats["classes"][label]
        z = np.abs(self.ewma[label] - s["mean"]) / s["std"]
        k = int(np.argmax(z))
        return float(z[k]), self.stats["names"][k]

    def update(self, ts, label, conf, features):
        if label not in self.ewma or conf < MIN_CONF or features[em.IDX["latest_W"]] < em.MIN_W:
            return []
        x = np.asarray(features, float)[self.cols]
        self.ewma[label] = (1 - self.alpha) * self.ewma[label] + self.alpha * x
        self.n[label] += 1
        if self.n[label] < int(2 / self.alpha):          # warm-up
            return []
        sc, name = self.score(label)
        if sc < self.limit or ts - self._last.get(label, -1e18) < COOLDOWN_S:
            return []
        self._last[label] = ts
        return [{"kind": "drift", "label": label, "ts": ts,
                 "message": f"{label}: {name} has drifted {sc:.1f} standard deviations from its usual level.",
                 "short": f"{label} drifting ({name})"}]

    def save(self, path="drift_state.json"):
        json.dump({"ewma": {c: v.tolist() for c, v in self.ewma.items()}, "n": self.n}, open(path, "w"))

    def load(self, path="drift_state.json"):
        try:
            d = json.load(open(path))
            self.ewma = {c: np.array(v) for c, v in d["ewma"].items() if c in self.ewma}
            self.n.update(d["n"])
        except (OSError, ValueError, KeyError):
            pass


def demo(csv, alpha, limit, ramp, ramp_windows, poll_s=7.0):
    X, y, _ = em.load_data(csv)
    stats = joblib.load("edge_stats.pkl")
    warm = int(2 / alpha)
    rows, clean_max = [], 0.0
    for c in stats["classes"]:
        Xc = X[y == c]
        reps = int(np.ceil(max(ramp_windows + warm, 400) / len(Xc))) + 1
        stream = np.tile(Xc, (reps, 1))

        mon = DriftMonitor(stats, alpha, 1e9)           # clean replay: worst score with no drift
        worst = 0.0
        for i, f in enumerate(stream[:max(400, warm * 3)]):
            mon.update(0, c, 1.0, f)
            if i >= warm:
                worst = max(worst, mon.score(c)[0])
        clean_max = max(clean_max, worst)

        mon = DriftMonitor(stats, alpha, 1e9)           # drifted replay: power creeps up by `ramp`
        hit = None
        for i, f in enumerate(stream[:ramp_windows + warm]):
            gain = 1 + ramp * min(i / ramp_windows, 1.0)
            mon.update(0, c, 1.0, em.scale_power(f[None, :], gain)[0])
            if i >= warm and hit is None and mon.score(c)[0] >= limit:
                hit = (i, gain)
        rows.append((c, worst, hit))

    print(f"alpha={alpha}, limit={limit}, simulated creep: +{ramp:.0%} power over {ramp_windows} windows\n")
    print(f"{'class':12s}{'clean max':>11s}   drift detected")
    for c, worst, hit in rows:
        txt = f"after {hit[0]} windows (~{hit[0] * poll_s / 60:.0f} min, at +{hit[1] - 1:.0%})" if hit else "NOT detected"
        print(f"{c:12s}{worst:11.2f}   {txt}")
    print(f"\nWorst clean score: {clean_max:.2f}. Pick a limit above it, e.g. --limit {clean_max * 1.2:.1f}. "
          "Clean scores above your limit mean false drift alarms.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv", default="training_data.csv")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--limit", type=float, default=LIMIT)
    ap.add_argument("--ramp", type=float, default=0.3)
    ap.add_argument("--ramp-windows", type=int, default=600)
    a = ap.parse_args()
    if not a.demo:
        raise SystemExit("Use --demo to test, or import DriftMonitor in webapp.py.")
    demo(a.csv, a.alpha, a.limit, a.ramp, a.ramp_windows)