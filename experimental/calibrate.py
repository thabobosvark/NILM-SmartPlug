"""
Calibration: ROC, threshold at the 99th percentile of normal scores, and
false alarms per day, for both anomaly methods.

  python calibrate.py --active-hours 4

Normal scores come from HELD-OUT data (whole sessions where a class has 2+).
Anomalies are generated from those same held-out windows:
  power_x1.5 / x2 / x0.5   current + power scaled together (surge / sag)
  unstable                 power swings 4x wider (arcing, loose connection)
  pf_shift                 power factor halved (reactive fault)
  cross_fed                features of class A presented as class B (labelled mode only)

Methods:
  mah_nearest   distance to the nearest class (label-free; what the ESP32 runs)
  mah_labelled  distance to the class the classifier named (server)
  ae            int8 autoencoder error (label-free; ESP32)

Outputs: edge_calib.json, roc_<method>.csv
"""
import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import edge_model as em
from autoencoder import QuantAE

IDX = em.IDX
METHODS = ("mah_nearest", "mah_labelled", "ae")
PERTURBED = ["power_x1.5", "power_x2", "power_x0.5", "unstable", "pf_shift"]


def make_anomaly(F, kind):
    if kind.startswith("power_x"):
        return em.scale_power(F, float(kind[7:]))
    G = np.array(F, float, copy=True)
    if kind == "unstable":
        mean = G[:, IDX["mean_W"]]
        G[:, IDX["std_W"]] *= 4.0
        G[:, IDX["range_W"]] *= 4.0
        G[:, IDX["max_W"]] = mean + G[:, IDX["range_W"]] / 2
        G[:, IDX["min_W"]] = np.maximum(mean - G[:, IDX["range_W"]] / 2, 0.0)
    elif kind == "pf_shift":
        G[:, [IDX["latest_PF"], IDX["mean_PF"]]] *= 0.5
    return G


def alarms(flags, groups, need):
    """Count runs of `need` flagged windows in a row inside a session (one alarm per run)."""
    n = 0
    for gid in np.unique(groups):
        run = 0
        for f in flags[groups == gid]:
            run = run + 1 if f else 0
            if run == need:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_data.csv")
    ap.add_argument("--active-hours", type=float, default=4.0, help="hours per day an appliance is actually on")
    ap.add_argument("--poll-s", type=float, default=7.0)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--pct", type=float, default=0.99)
    a = ap.parse_args()

    X, y, g = em.load_data(a.csv)
    cols = joblib.load("edge_stats.pkl")["cols"]
    tr, va = em.split_masks(y, g)
    stats = em.fit(X[tr], y[tr], cols)              # trained WITHOUT the validation windows
    ok = va & np.isin(y, list(stats["classes"]))
    Xv, yv, gv = X[ok], y[ok], g[ok]
    print(f"{len(Xv)} held-out normal windows from {len(set(yv))} classes")
    if len(stats["classes"]) < 2:
        raise SystemExit("Need at least 2 classes.")

    ae = QuantAE()

    def score(F, claimed):
        return {"mah_nearest": em.ratios(stats, F)[0],
                "mah_labelled": em.ratios(stats, F, claimed)[0],
                "ae": ae.errors(F[:, cols])}

    normal = score(Xv, yv)
    anomalies = {}
    for kind in PERTURBED:
        G = make_anomaly(Xv, kind)
        keep = G[:, IDX["latest_W"]] >= em.MIN_W     # the ESP32 would not score these
        anomalies[kind] = score(G[keep], yv[keep])

    rng = np.random.default_rng(0)
    names = list(stats["classes"])
    other = np.array([rng.choice([n for n in names if n != c]) for c in yv])
    anomalies["cross_fed"] = {"mah_labelled": em.ratios(stats, Xv, other)[0]}

    print("\nAUC (1.0 = perfect, 0.5 = coin flip)")
    print(f"{'anomaly':12s}" + "".join(f"{m:>14s}" for m in METHODS))
    for kind, sc in anomalies.items():
        row = ""
        for m in METHODS:
            if m in sc:
                lab = np.r_[np.zeros(len(normal[m])), np.ones(len(sc[m]))]
                row += f"{roc_auc_score(lab, np.r_[normal[m], sc[m]]):14.3f}"
            else:
                row += f"{'n/a':>14s}"
        print(f"{kind:12s}{row}")

    wpd = a.active_hours * 3600 / a.poll_s
    calib = {"pct": a.pct, "confirm": a.confirm, "poll_s": a.poll_s, "active_hours": a.active_hours,
             "n_val": int(len(Xv))}
    print(f"\nThreshold = {a.pct:.1%} percentile of normal scores. Assumes {a.active_hours} h/day active, "
          f"{a.poll_s:.0f} s polling ({wpd:.0f} windows/day)")
    for m in METHODS:
        thr = float(np.quantile(normal[m], a.pct))
        flags = normal[m] > thr
        fpd_raw = flags.mean() * wpd
        fpd_conf = alarms(flags, gv, a.confirm) / len(flags) * wpd
        print(f"\n{m}: threshold {thr:.4f}")
        print(f"  false alarms/day: {fpd_raw:.1f} single-window, {fpd_conf:.2f} with {a.confirm}-in-a-row")
        for kind, sc in anomalies.items():
            if m in sc:
                print(f"  detected {kind:11s} {np.mean(sc[m] > thr):6.1%}")
        allp = np.concatenate([anomalies[k][m] for k in PERTURBED])
        lab = np.r_[np.zeros(len(normal[m])), np.ones(len(allp))]
        fpr, tpr, th = roc_curve(lab, np.r_[normal[m], allp])
        np.savetxt(f"roc_{m}.csv", np.c_[fpr, tpr, th], delimiter=",", header="fpr,tpr,threshold", comments="")
        calib[m + "_thr"] = thr

    calib["mah_limit"] = calib["mah_nearest_thr"]
    calib["ae_limit"] = calib["ae_thr"]
    json.dump(calib, open("edge_calib.json", "w"), indent=1)
    print("\nSaved edge_calib.json and roc_*.csv")


if __name__ == "__main__":
    main()