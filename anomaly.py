"""
Anomaly detection: flags readings that are abnormal for the appliance that is
running, or abnormal for the mains supply.

Three checks, all explainable (no black box):

  level    A recognised appliance draws far more (or less) power than it did in
           your training data. The normal band for each appliance is its 1st to
           99th percentile of watts, widened by 25% or 2 W (whichever is larger). Three readings in a row
           outside the band are needed, so a single spike is ignored.
  supply   Mains voltage outside 207-253 V (230 V +/-10%) or frequency outside
           49-51 Hz, three readings in a row.
  overload Power above OVERLOAD_W, two readings in a row.

Each (kind, appliance) alert has a cooldown so a long fault is reported once
per COOLDOWN_S, not every 7 seconds.

Profiles come from training_data.csv, so a different charger or a faulty
appliance that behaves unlike your recordings will be flagged.
"""
import math
from collections import defaultdict

import pandas as pd

V_MIN, V_MAX = 207.0, 253.0
HZ_MIN, HZ_MAX = 49.0, 51.0
OVERLOAD_W = 2400.0
CONFIRM = 3
COOLDOWN_S = 900
MIN_CONF = 0.7
BAND = 1.25      # 25% tolerance beyond the usual range
TOL_W = 2.0      # ...but never less than 2 W
SKIP = {"nothing_plugged", "warming_up", "unknown"}


def build_profiles(csv_path):
    """{label: {"lo","hi","median","n"}} from the labelled training data."""
    try:
        df = pd.read_csv(csv_path, usecols=["label", "W"])
    except Exception:
        return {}
    df["W"] = pd.to_numeric(df["W"], errors="coerce")
    df = df.dropna()
    df = df[df["W"] >= 1.0]            # ignore thermostat-off zeros
    out = {}
    for lab, g in df.groupby("label"):
        if lab in SKIP or len(g) < 20:
            continue
        lo, med, hi = g["W"].quantile([0.01, 0.5, 0.99])
        # allow max(2 W, 25%) beyond the usual range, so a 20 W charger may reach 22 W or more
        out[str(lab)] = {
            "lo": float(max(lo - max(TOL_W, (BAND - 1.0) * lo), 0.0)),
            "hi": float(hi + max(TOL_W, (BAND - 1.0) * hi)),
            "median": float(med),
            "n": int(len(g)),
        }
    return out


class AnomalyDetector:
    def __init__(self, profiles):
        self.profiles = profiles or {}
        self._streak = defaultdict(int)
        self._last = {}

    def set_profiles(self, profiles):
        self.profiles = profiles or {}

    def _hit(self, key, cond, ts, kind, label, message, short, out):
        """Count consecutive hits; emit once CONFIRM is reached and the cooldown allows."""
        if not cond:
            self._streak[key] = 0
            return
        self._streak[key] += 1
        need = 2 if kind == "overload" else CONFIRM
        if self._streak[key] < need:
            return
        if ts - self._last.get(key, 0.0) < COOLDOWN_S:
            return
        self._last[key] = ts
        out.append({"kind": kind, "label": label, "message": message, "short": short, "ts": ts})

    def update(self, ts, label, conf, reading):
        out = []
        v, hz, w = reading["V"], reading.get("Hz", 0.0), reading["W"]

        self._hit(("supply", "V"), v > 0 and not (V_MIN <= v <= V_MAX), ts, "supply", None,
                  f"Supply voltage {v:.0f} V is outside the safe range ({V_MIN:.0f} to {V_MAX:.0f} V).",
                  f"Voltage {v:.0f} V", out)
        self._hit(("supply", "Hz"), hz > 0 and not (HZ_MIN <= hz <= HZ_MAX), ts, "supply", None,
                  f"Supply frequency {hz:.1f} Hz is outside {HZ_MIN:.0f} to {HZ_MAX:.0f} Hz.",
                  f"Frequency {hz:.1f} Hz", out)
        self._hit(("overload", None), w > OVERLOAD_W, ts, "overload", None,
                  f"Load of {w:.0f} W is above the {OVERLOAD_W:.0f} W overload limit.",
                  f"Overload {w:.0f} W", out)

        prof = self.profiles.get(label)
        if prof and conf >= MIN_CONF and w >= 1.0 and math.isfinite(w):
            too_high, too_low = w > prof["hi"], w < prof["lo"]
            name = label
            self._hit(("level", label), too_high or too_low, ts, "level", label,
                      f"{name} is drawing {w:.0f} W, normally about {prof['median']:.0f} W "
                      f"(usual range {prof['lo']:.0f} to {prof['hi']:.0f} W).",
                      f"{name} {w:.0f} W abnormal", out)
        else:
            for k in [k for k in self._streak if k[0] == "level"]:
                if label != k[1]:
                    self._streak[k] = 0
        return out
