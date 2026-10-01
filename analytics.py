"""
Per-appliance power, current, energy, cost, CO2 and standby analysis, computed
from live_log.csv.

Energy = watts x seconds between consecutive readings. Gaps longer than
max_gap seconds (sensor offline) are not counted.

"Standby" here means a stretch where the load sits between STANDBY_MIN_W and
STANDBY_MAX_W for at least STANDBY_MIN_MINUTES. Edit the three constants
below if your idle noise or charger trickle looks different.
"""
from datetime import datetime, timedelta

import pandas as pd

IDLE = {"nothing_plugged", "warming_up"}

STANDBY_MIN_W = 0.3
STANDBY_MAX_W = 4.0
STANDBY_MIN_MINUTES = 30


def _since(rng):
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if rng == "30d":
        return midnight - timedelta(days=29)
    if rng == "7d":
        return midnight - timedelta(days=6)
    return midnight


def _load(path, since):
    try:
        df = pd.read_csv(path, usecols=["time", "prediction", "confidence", "I", "W"], on_bad_lines="skip")
    except Exception:
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("confidence", "I", "W"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time")
    df = df[df["time"] >= since].reset_index(drop=True)
    return df if len(df) else None


def _intervals(df, max_gap):
    """Seconds each reading 'covers' (until the next reading), 0 across gaps."""
    sec = (df["time"].shift(-1) - df["time"]).dt.total_seconds()
    return sec.where((sec > 0) & (sec <= max_gap), 0.0).fillna(0.0)


def _energy(df, tariff, co2, unknown_conf, max_gap):
    sec = _intervals(df, max_gap)
    label = df["prediction"].where(
        df["prediction"].isin(IDLE) | (df["confidence"] >= unknown_conf), "unknown"
    )
    label = label.mask(label == "warming_up").ffill().fillna("nothing_plugged")
    kwh = df["W"] * sec / 3.6e6
    d = pd.DataFrame({
        "label": label, "kwh": kwh, "sec": sec,
        "w_sec": df["W"] * sec, "a_sec": df["I"] * sec,
        "w": df["W"], "a": df["I"],
    })
    g = d.groupby("label").agg(
        kwh=("kwh", "sum"), sec=("sec", "sum"), w_sec=("w_sec", "sum"),
        a_sec=("a_sec", "sum"), peak_w=("w", "max"), peak_a=("a", "max"),
    )

    total = float(g["kwh"].sum())
    rows = []
    for lab, r in g.sort_values("kwh", ascending=False).iterrows():
        if lab == "nothing_plugged":
            continue
        secs = float(r["sec"])
        rows.append({
            "label": lab,
            "avg_w": round(float(r["w_sec"]) / secs, 1) if secs > 0 else 0.0,
            "peak_w": round(float(r["peak_w"]), 1),
            "avg_a": round(float(r["a_sec"]) / secs, 3) if secs > 0 else 0.0,
            "peak_a": round(float(r["peak_a"]), 3),
            "kwh": round(float(r["kwh"]), 4),
            "cost": round(float(r["kwh"]) * tariff, 2),
            "co2": round(float(r["kwh"]) * co2, 3),
            "seconds": int(secs),
        })
    totals = {"kwh": round(total, 4), "cost": round(total * tariff, 2), "co2": round(total * co2, 3)}
    return rows, totals


def _standby(df, tariff, max_gap):
    sec = _intervals(df, max_gap)
    mask = (df["W"] >= STANDBY_MIN_W) & (df["W"] < STANDBY_MAX_W)
    gap = df["time"].diff().dt.total_seconds().fillna(0)
    run_id = ((~mask) | (gap > 60)).cumsum()

    runs = []
    for _, g in df[mask].groupby(run_id[mask]):
        s = sec.loc[g.index]
        minutes = float(s.sum()) / 60.0
        if minutes < STANDBY_MIN_MINUTES:
            continue
        kwh = float((g["W"] * s).sum()) / 3.6e6
        runs.append({
            "ts": g["time"].iloc[0].timestamp(),
            "start": g["time"].iloc[0].strftime("%d %b %H:%M"),
            "minutes": round(minutes, 1),
            "avg_w": round(float(g["W"].mean()), 2),
            "kwh": round(kwh, 4),
        })
    runs.sort(key=lambda r: r["ts"], reverse=True)
    total_kwh = sum(r["kwh"] for r in runs)
    return {
        "runs": runs[:5],
        "count": len(runs),
        "minutes": round(sum(r["minutes"] for r in runs), 1),
        "kwh": round(total_kwh, 4),
        "cost": round(total_kwh * tariff, 2),
    }


def analyze(path, rng, tariff, co2, unknown_conf=0.5, max_gap=30):
    empty = {
        "rows": [], "totals": {"kwh": 0, "cost": 0, "co2": 0},
        "standby": {"runs": [], "count": 0, "minutes": 0, "kwh": 0, "cost": 0},
        "tariff": tariff, "co2_factor": co2,
    }
    df = _load(path, _since(rng))
    if df is None:
        return empty
    rows, totals = _energy(df, tariff, co2, unknown_conf, max_gap)
    return {
        "rows": rows, "totals": totals,
        "standby": _standby(df, tariff, max_gap),
        "tariff": tariff, "co2_factor": co2,
    }