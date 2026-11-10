"""
Open-set demo for your report: pretend one appliance was never recorded, then
see whether the system calls it "unknown".

    python demo_openset.py              # holds out blender, then iPad
    python demo_openset.py --hold blender

For each held-out appliance it reports
  caught as unknown   share of its windows the open-set check flags (higher is better)
  known false alarms  share of windows from known appliances wrongly flagged,
                      measured by leaving one whole capture session out (lower is better)
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import openset
from train_model import assign_sessions, build_windowed_dataset

warnings.filterwarnings("ignore")


def flagged(stats, X):
    return np.array([openset.score(stats, x)[0] > 1.0 for x in X])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_data.csv")
    ap.add_argument("--hold", nargs="*", default=["blender", "iPad"])
    args = ap.parse_args()

    df = assign_sessions(pd.read_csv(args.csv))
    X, y, g = build_windowed_dataset(df)
    base = y != "nothing_plugged"

    # False alarms on known appliances: leave one session out, fit on the rest.
    fa, n = 0, 0
    for s in np.unique(g[base]):
        c = y[g == s][0]
        if len(np.unique(g[base & (y == c)])) < 2:
            continue                     # the only session of its class: nothing to fall back on
        tr = base & (g != s)
        stats = openset.fit(X[tr], y[tr])
        f = flagged(stats, X[g == s])
        fa += int(f.sum())
        n += len(f)
    print(f"Known-appliance false alarms (leave-one-session-out): {fa / max(n, 1):.1%} of {n} windows\n")

    for held in args.hold:
        if held not in set(y):
            print(f"{held}: not in the data, skipped")
            continue
        keep = base & (y != held)
        stats = openset.fit(X[keep], y[keep])
        f = flagged(stats, X[y == held])
        near = [openset.score(stats, x)[1] for x in X[y == held]]
        nearest = pd.Series(near).value_counts().idxmax()
        print(f"{held:8s} held out -> caught as unknown: {f.mean():.0%} of {len(f)} windows "
              f"(looks most like: {nearest})")


if __name__ == "__main__":
    main()
