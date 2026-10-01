"""
Runs the live classifier, event detector, open-set check, anomaly detector,
analytics, email alerts, the human-in-the-loop labelling flow AND the web
dashboard in one process.

Run: python webapp.py --broker 192.168.10.241
Then open http://localhost:8000 in a browser.

Remote access (phone on the same Wi-Fi) is OFF until you set a password:
  DASHBOARD_PASSWORD=choose-something-long     (user name: admin)
Without it the dashboard only answers on this laptop (localhost).

Settings live in the .env file next to this script (loaded automatically).
Real environment variables set in the terminal win over the file.

Optional settings
  Email alerts   SMTP_USER, SMTP_APP_PASSWORD, ALERT_EMAIL_TO
  Dashboard      DASHBOARD_USER (default admin), DASHBOARD_PASSWORD
  MQTT broker    MQTT_BROKER, MQTT_PORT (default 1883; use 8883 with TLS),
                 MQTT_USER, MQTT_PASSWORD, MQTT_TLS_CA (path to ca.crt)

Files this needs in the same folder:
  features.py, nilm_model.pkl, openset.py, anomaly.py, train_model.py,
  events.py, store.py, analytics.py, alerts.py, training_data.csv,
  static/dashboard.html
(openset.pkl is created automatically if it is missing.)

Every classification is appended to live_log.csv. Detected events, alerts,
anomalies and your labelling decisions are kept in smartplug.db, which is
snapshotted daily into db_backups/.
"""
import argparse
import csv
import hmac
import json
import math
import os
import re
import shutil
import time
import threading
from collections import Counter, deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, request, Response

import anomaly
import openset
import train_model
from features import (
    extract_features, WINDOW_SIZE, FEATURE_NAMES,
    ZERO_W_THRESHOLD, ZERO_I_THRESHOLD, IRON_HOLD_SECONDS,
)
from events import EventDetector
from store import EventStore
import analytics
from alerts import AlertEngine, RULES, describe, send_alert, channels_configured

# Load .env from this script's folder, before any setting below is read.
# Terminal environment variables win over the file (override=False).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass   # python-dotenv not installed: real environment variables only

# ---------- Config ----------
LOG_FILE = "live_log.csv"
LOG_FIELDS = ["time", "device_id", "prediction", "confidence", "V", "I", "W", "PF"]
DB_FILE = os.environ.get("SMARTPLUG_DB", "smartplug.db")
TRAIN_CSV = "training_data.csv"
MODEL_FILE = "nilm_model.pkl"
OPENSET_FILE = "openset.pkl"
MODEL_BACKUP_DIR = "model_backups"
DB_BACKUP_DIR = "db_backups"

TARIFF_R_PER_KWH = 4.1379   # Cape Town Domestic Block 1 (2026/27, incl. VAT). Check against your own bill.
CO2_KG_PER_KWH = 0.985      # DFFE national generation grid emission factor (2021 data). Check the latest DFFE report.
UNKNOWN_CONF = 0.5          # below this confidence, a non-idle reading is shown as "unknown"
MAX_GAP_S = 30              # gaps longer than this are not counted as energy (sensor was offline)
STALE_S = 20                # no reading for this long = dashboard shows "No data"
ALERT_CHECK_S = 5           # how often alert rules are evaluated
OFFLINE_S = 120             # no reading for this long = "sensor offline" alert
RETRAIN_MIN_ACC = 0.85      # a retrained model is only adopted if its held-out accuracy is at least this
REVIEW_DAYS = 30            # how far back the "please label this" queue looks

# Live teaching: while something is running and SmartPlug cannot identify it,
# ask the person to name it once it has been "unknown" for this long.
UNKNOWN_PROMPT_S = 30
RUN_END_GAP_S = 20          # no load for this long = the run is over
RUN_MIN_W = 1.0             # a reading at or above this counts as "something is running"
RUN_MAX_ROWS = 2000
HELD_RUN_S = 600            # a finished unknown run stays nameable for this long

# Automatic data collection. Only runs that were recognised cleanly are kept,
# and they go to auto_data.csv (NOT training_data.csv). Delete that file to undo.
AUTO_COLLECT = True
AUTO_CSV = "auto_data.csv"
AUTO_MIN_CONF = 0.90        # average confidence of the whole run
AUTO_PURITY = 0.98          # share of readings that must carry the same label
AUTO_MAX_RATIO = 0.8        # open-set ratio must stay well inside "known" (1.0 = unknown)
AUTO_MIN_S = 60             # ignore very short runs
AUTO_MAX_ROWS_PER_LABEL_DAY = 600

DASH_USER = os.environ.get("DASHBOARD_USER", "admin")
DASH_PASS = os.environ.get("DASHBOARD_PASSWORD", "")
LOCAL_IPS = {"127.0.0.1", "::1"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}   # Host header values accepted in localhost-only mode

# physically plausible ranges; anything else is a sensor/communication fault
LIMITS = {"V": (0, 320), "I": (0, 100), "W": (0, 25000), "PF": (-0.01, 1.05)}

# ---------- State ----------
models = {"clf": None, "openset": None}
model_lock = threading.Lock()
data_lock = threading.Lock()          # guards training_data.csv
retrain_state = {"running": False, "last": None}

device_windows = {}
device_last_active = {}
history = deque(maxlen=50)
latest_state = {"prediction": None, "confidence": None, "reading": None, "updated": None, "ts": None}
state_lock = threading.Lock()
health = {"start": time.time(), "mqtt": False, "messages": 0, "rejected": 0,
          "offline_alerted": False, "last_reject_print": 0.0}

store = EventStore(DB_FILE)
detector = EventDetector()
engine = AlertEngine(RULES)
anomalies = anomaly.AnomalyDetector({})


def load_models():
    """(Re)load the classifier and the open-set statistics. Builds the latter if missing."""
    clf = joblib.load(MODEL_FILE)
    stats = None
    if os.path.isfile(OPENSET_FILE):
        try:
            stats = joblib.load(OPENSET_FILE)
        except Exception as e:
            print("Could not read openset.pkl:", e)
    if stats is None:
        try:
            stats = train_model.fit_openset_from_csv(TRAIN_CSV, OPENSET_FILE)
            print("Built openset.pkl from training_data.csv")
        except Exception as e:
            print("Open-set check disabled:", e)
    with model_lock:
        models["clf"], models["openset"] = clf, stats
    anomalies.set_profiles(anomaly.build_profiles(TRAIN_CSV))


def to_display(pred, conf):
    """Turn a raw model label into what the dashboard shows."""
    if pred in ("nothing_plugged", "warming_up"):
        return pred
    if conf < UNKNOWN_CONF:
        return "unknown"
    return pred


class Tracker:
    """Keeps today's energy, peak load and the top appliance by energy."""

    def __init__(self):
        self.day = None
        self.day_kwh = defaultdict(float)
        self.by_label = {}
        self.peak_w = 0.0
        self.peak_ts = None
        self.last_ts = None
        self.last_label = None

    def close_open(self):
        self.last_ts = None

    def update(self, ts, label, w):
        d = datetime.fromtimestamp(ts).date()
        if d != self.day:
            self.day = d
            self.by_label = {}
            self.peak_w = 0.0
            self.peak_ts = None

        # Energy: W x seconds, skipping long gaps
        if self.last_ts is not None:
            dt = ts - self.last_ts
            if 0 < dt <= MAX_GAP_S:
                kwh = w * dt / 3.6e6
                self.day_kwh[d] += kwh
                key = self.last_label if (label == "warming_up" and self.last_label) else label
                self.by_label[key] = self.by_label.get(key, 0.0) + kwh
        self.last_ts = ts
        if label != "warming_up":
            self.last_label = label

        if w > self.peak_w:
            self.peak_w = w
            self.peak_ts = ts

    def snapshot(self, now):
        today = datetime.fromtimestamp(now).date()
        same = (self.day == today)
        kwh = self.day_kwh.get(today, 0.0)
        yesterday = self.day_kwh.get(today - timedelta(days=1))

        by = self.by_label if same else {}
        cand = {k: v for k, v in by.items() if k not in ("nothing_plugged", "warming_up") and v > 0}
        top = None
        if cand:
            k = max(cand, key=cand.get)
            top = {"label": k, "kwh": round(cand[k], 4)}

        return {
            "kwh": round(kwh, 4),
            "yesterday_kwh": round(yesterday, 4) if yesterday is not None else None,
            "peak_w": round(self.peak_w, 1) if same else 0.0,
            "peak_time": datetime.fromtimestamp(self.peak_ts).strftime("%H:%M") if (same and self.peak_ts) else None,
            "top": top,
        }


tracker = Tracker()


class RunTracker:
    """Follows one 'run' (from the load appearing to it going away) so SmartPlug
    can ask about an unknown appliance while it is still running, and can keep
    clean, confident runs as extra training data."""

    def __init__(self):
        self.lock = threading.Lock()
        self.run = None
        self.held = None          # (run, finished_at): ended with an unknown stretch, still unnamed
        self.handled = set()      # run ids the person named or dismissed
        self.auto_rows = {}       # (date, label) -> rows auto-collected

    @staticmethod
    def _new(ts):
        return {"id": int(ts), "start": ts, "last_active": ts, "last_ts": None, "rows": [],
                "labels": Counter(), "conf_sum": 0.0, "conf_n": 0, "max_ratio": 0.0,
                "unknown_s": 0.0, "teach_label": None, "saved_upto": 0}

    def _close(self):
        run, self.run = self.run, None
        while run["rows"] and run["rows"][-1][3] < RUN_MIN_W:
            run["rows"].pop()
        if (run["unknown_s"] >= UNKNOWN_PROMPT_S and run["id"] not in self.handled
                and not run["teach_label"]):
            self.held = (run, time.time())
        return run

    def update(self, ts, r, label, conf, ratio):
        """Feed one reading. Returns a run if one has just finished."""
        finished = None
        with self.lock:
            run = self.run
            if run and ts - run["last_active"] > RUN_END_GAP_S:
                finished = self._close()
                run = None
            active = r["W"] >= RUN_MIN_W
            if run is None:
                if not active:
                    return finished
                run = self.run = self._new(ts)
            dt = min(ts - run["last_ts"], MAX_GAP_S) if run["last_ts"] is not None else 0.0
            run["last_ts"] = ts
            if active:
                run["last_active"] = ts
                if label != "warming_up":
                    run["labels"][label] += 1
                    if label == "unknown":
                        run["unknown_s"] += dt
                    else:
                        run["conf_sum"] += conf
                        run["conf_n"] += 1
                        run["max_ratio"] = max(run["max_ratio"], ratio)
            if len(run["rows"]) < RUN_MAX_ROWS:
                run["rows"].append((ts, r["V"], r["I"], r["W"], r["PF"]))
        return finished

    def prompt(self, now):
        """The unknown run SmartPlug should be asking about right now, or None."""
        with self.lock:
            run, running = self.run, True
            if not (run and now - run["last_active"] <= RUN_END_GAP_S):
                run, running = None, False
                if self.held and now - self.held[1] < HELD_RUN_S:
                    run = self.held[0]
            if (not run or run["id"] in self.handled or run["teach_label"]
                    or run["unknown_s"] < UNKNOWN_PROMPT_S):
                return None
            recent = [x[3] for x in run["rows"][-10:] if x[3] >= RUN_MIN_W]
            return {"run_id": run["id"], "running": running,
                    "seconds": int(run["last_active"] - run["start"]),
                    "watts": round(sum(recent) / len(recent), 1) if recent else 0.0}

    def dismiss(self, run_id):
        with self.lock:
            self.handled.add(run_id)
            if self.held and self.held[0]["id"] == run_id:
                self.held = None

    def name_run(self, run_id, label):
        """Save the run so far as training data. Returns (ok, detail, already_finished)."""
        with self.lock:
            run, finished = None, False
            if self.run and self.run["id"] == run_id:
                run = self.run
            elif self.held and self.held[0]["id"] == run_id:
                run, finished = self.held[0], True
            if run is None:
                return False, "That run is no longer available.", False
            rows = list(run["rows"])
            while rows and rows[-1][3] < RUN_MIN_W:
                rows.pop()
            if len(rows) < WINDOW_SIZE + 3:
                return False, f"Only {len(rows)} readings so far. Give it a little longer.", False
            append_rows(TRAIN_CSV, rows, label)
            run["teach_label"] = label
            run["saved_upto"] = len(run["rows"])
            self.handled.add(run_id)
            if finished:
                self.held = None
            return True, f"Saved {len(rows)} readings as '{label}'.", finished


runs = RunTracker()


def append_rows(path, rows, label):
    """Append (ts, V, I, W, PF) rows to a training-format CSV."""
    with data_lock:
        new_file = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "label", "V", "I", "W", "PF", "Hz", "kWh"])
            for ts, v, i, watts, pf in rows:
                w.writerow([ts, label, v, i, watts, pf, 50.0, 0.0])   # Hz and kWh are not used by the model


def start_retrain():
    if retrain_state["running"]:
        return False
    retrain_state["running"] = True
    threading.Thread(target=_retrain_job, daemon=True).start()
    return True


def run_is_clean(run):
    """True only if the whole run was recognised as ONE known appliance, confidently."""
    if run["labels"].get("unknown") or run["unknown_s"] > 0:
        return False
    if len(run["rows"]) < WINDOW_SIZE + 3 or run["last_active"] - run["start"] < AUTO_MIN_S:
        return False
    n = sum(run["labels"].values())
    if not n:
        return False
    label, cnt = run["labels"].most_common(1)[0]
    if label in RESERVED or label not in known_classes() or cnt / n < AUTO_PURITY:
        return False
    if run["conf_n"] == 0 or run["conf_sum"] / run["conf_n"] < AUTO_MIN_CONF:
        return False
    return run["max_ratio"] <= AUTO_MAX_RATIO


def handle_finished_run(run):
    try:
        if run["teach_label"]:
            # Named while it was running: add the rest of the run, then learn it.
            extra = run["rows"][run["saved_upto"]:]
            if extra:
                append_rows(TRAIN_CSV, extra, run["teach_label"])
            start_retrain()
        elif AUTO_COLLECT and run["id"] not in runs.handled and run_is_clean(run):
            label = run["labels"].most_common(1)[0][0]
            key = (datetime.fromtimestamp(run["start"]).date(), label)
            room = AUTO_MAX_ROWS_PER_LABEL_DAY - runs.auto_rows.get(key, 0)
            if room > 0:
                rows = run["rows"][:room]
                append_rows(AUTO_CSV, rows, label)
                runs.auto_rows[key] = runs.auto_rows.get(key, 0) + len(rows)
                print(f"Auto-collected {len(rows)} readings of '{label}' into {AUTO_CSV}")
    except Exception as e:
        print("Run post-processing failed:", e)


def feed_detector(ts, w, pf, label, conf):
    """Run one reading through the event detector and persist the result."""
    emitted, updated = detector.update(ts, w, pf, label, conf)
    for ev in emitted:
        ev["id"] = store.insert(ev)
    for ev in updated:
        store.update(ev)


def replay_log():
    """Rebuild today's numbers from live_log.csv; back-fill events if the DB is empty."""
    if not os.path.isfile(LOG_FILE):
        return
    backfill = store.count() == 0
    count = 0
    try:
        with open(LOG_FILE, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S").timestamp()
                    conf = float(row["confidence"])
                    w = float(row["W"])
                    pf = float(row["PF"])
                    label = to_display(row["prediction"], conf)
                except (KeyError, ValueError, TypeError):
                    continue
                tracker.update(ts, label, w)
                if backfill:
                    feed_detector(ts, w, pf, label, conf)
                count += 1
    except Exception as e:
        print("Could not replay live_log.csv:", e)
    tracker.close_open()
    detector.reset(resumed=True)
    print(f"Rebuilt state from {count} logged readings"
          f"{' and back-filled the event timeline' if backfill else ''}.")


def log_row(device_id, pred, confidence, reading):
    """Append one classification to live_log.csv (never lets a log error kill MQTT)."""
    try:
        new_file = not os.path.isfile(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": device_id,
                "prediction": pred,
                "confidence": round(confidence, 3),
                "V": reading["V"], "I": reading["I"],
                "W": reading["W"], "PF": reading["PF"],
            })
    except Exception as e:
        print("live_log write failed:", e)


def valid_reading(r):
    """Reject NaN/infinite or physically impossible values (a faulty sensor or garbled message)."""
    try:
        if not all(math.isfinite(v) for v in r.values()):
            return False
        for k, (lo, hi) in LIMITS.items():
            if not (lo <= r[k] <= hi):
                return False
        return r["Hz"] == 0 or 40 <= r["Hz"] <= 70
    except (KeyError, TypeError):
        return False


def classify(device_id, reading):
    """Returns (prediction, confidence, openset_ratio). ratio > 1 means 'not like any known appliance'."""
    if device_id not in device_windows:
        device_windows[device_id] = deque(maxlen=WINDOW_SIZE)
    device_windows[device_id].append(reading)

    now = time.time()
    is_zero = reading["W"] < ZERO_W_THRESHOLD and reading["I"] < ZERO_I_THRESHOLD
    window = list(device_windows[device_id])
    ratio = 0.0

    if is_zero:
        # Zero power has one correct answer, so no ML needed. Exception:
        # the iron's thermostat dips also read as zero, so if iron was the
        # last confident prediction, hold "iron" for IRON_HOLD_SECONDS.
        last = device_last_active.get(device_id)
        if last and last["label"] == "iron" and (now - last["time"]) < IRON_HOLD_SECONDS:
            pred, confidence = "iron", 1.0
        else:
            pred, confidence = "nothing_plugged", 1.0
    elif len(window) < WINDOW_SIZE:
        # The model was trained only on full WINDOW_SIZE windows.
        pred, confidence = "warming_up", 0.0
    else:
        feats = extract_features(window)
        X = pd.DataFrame([feats], columns=FEATURE_NAMES)
        with model_lock:
            clf, stats = models["clf"], models["openset"]
        pred = str(clf.predict(X)[0])
        confidence = float(max(clf.predict_proba(X)[0]))
        if pred != "nothing_plugged":
            if stats is not None:
                ratio, _ = openset.score(stats, feats)
            if ratio > 1.0:
                pred, confidence = "unknown", 0.0     # looks unlike every appliance we know
            else:
                device_last_active[device_id] = {"label": pred, "time": now}

    return pred, confidence, ratio


# ---------- MQTT ----------
def on_connect(client, userdata, flags, rc, properties=None):
    health["mqtt"] = (rc == 0)
    print("Connected to MQTT broker" if rc == 0
          else f"MQTT connect failed, rc={rc} (check MQTT_USER, MQTT_PASSWORD, MQTT_TLS_CA and the port)")
    client.subscribe("nilm/plug1/reading")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    health["mqtt"] = False
    print("MQTT disconnected, retrying automatically...")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if not isinstance(payload, dict):
            return
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    try:
        device_id = str(payload.get("device_id", "plug1"))[:32]
        reading = {k: float(payload.get(k, 0.0)) for k in ["V", "I", "W", "PF", "Hz", "kWh"]}
    except (TypeError, ValueError):
        health["rejected"] += 1
        return
    if not valid_reading(reading):
        health["rejected"] += 1
        if time.time() - health["last_reject_print"] > 60:
            health["last_reject_print"] = time.time()
            print("Rejected an invalid sensor reading (NaN or out of range). Check the PZEM wiring.")
        return

    try:
        health["messages"] += 1
        raw_pred, confidence, ratio = classify(device_id, reading)
        label = to_display(raw_pred, confidence)
        now = time.time()

        entry = {
            "time": time.strftime("%H:%M:%S"),
            "prediction": label,
            "confidence": round(confidence, 3),
            "W": reading["W"],
            "PF": reading["PF"],
        }

        with state_lock:
            tracker.update(now, label, reading["W"])
            feed_detector(now, reading["W"], reading["PF"], label, confidence)
            history.append(entry)
            latest_state.update({
                "prediction": label,
                "confidence": round(confidence, 3),
                "reading": reading,
                "updated": entry["time"],
                "ts": now,
            })

        finished = runs.update(now, reading, label, confidence, ratio)
        if finished:
            handle_finished_run(finished)

        for a in anomalies.update(now, label, confidence, reading):
            print("ANOMALY:", a["message"])
            deliver("ANOMALY", a["kind"], a["label"], a["message"], a["short"], a["ts"])

        client.publish("nilm/plug1/result", json.dumps(
            {"device_id": device_id, "prediction": label, "confidence": round(confidence, 3)}
        ))
        log_row(device_id, raw_pred, confidence, reading)
        print(f"[{entry['time']}] {label} (raw={raw_pred}, conf={confidence:.2f}) "
              f"W={reading['W']:.1f} I={reading['I']:.3f} PF={reading['PF']:.2f}")
    except Exception as e:
        print("Error processing message:", e)


def start_mqtt(broker, port):
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    user = os.environ.get("MQTT_USER")
    if user:
        client.username_pw_set(user, os.environ.get("MQTT_PASSWORD", ""))
    ca = os.environ.get("MQTT_TLS_CA")
    if ca:
        client.tls_set(ca_certs=ca)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect_async(broker, port, keepalive=60)
    client.loop_forever(retry_first_connection=True)   # keeps retrying if the broker is down


# ---------- Alerts, anomalies, watchdog ----------
def deliver(ev_type, rule, label, message, short, ts):
    """Record in the timeline and send an email in the background."""
    try:
        store.insert({"ts": ts, "type": ev_type, "label": label, "note": message})
    except Exception as e:
        print("Could not store alert:", e)

    def _send():
        ok, detail = send_alert("SmartPlug: " + message, short=short)
        print(f"{ev_type} '{rule}' delivery -> {detail}")

    threading.Thread(target=_send, daemon=True).start()


def alert_loop():
    while True:
        time.sleep(ALERT_CHECK_S)
        try:
            now = time.time()
            with state_lock:
                st = detector.status()
                ts = latest_state.get("ts")
            fresh = ts is not None and (now - ts) < STALE_S
            for a in engine.check(now, st, fresh):
                print("ALERT:", a["message"])
                deliver("ALERT", a["rule"], a["label"], a["message"], a.get("short"), a["ts"])

            # Watchdog: the sensor going silent is a fault in its own right.
            last = ts if ts is not None else health["start"]
            if now - last > OFFLINE_S and not health["offline_alerted"]:
                health["offline_alerted"] = True
                mins = int((now - last) // 60)
                deliver("ALERT", "sensor_offline", None,
                        f"No readings from the sensor for {mins} min.", "Sensor offline", now)
            elif fresh and health["offline_alerted"]:
                health["offline_alerted"] = False
                deliver("ALERT", "sensor_online", None, "Sensor is reporting again.", "Sensor back online", now)
        except Exception as e:
            print("alert loop error:", e)


# ---------- Active learning: label runs, retrain ----------
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
RESERVED = {"nothing_plugged", "warming_up", "unknown"}


def readings_between(t0, t1):
    out = []
    with open(LOG_FILE, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S").timestamp()
                if ts < t0 or ts > t1:
                    continue
                out.append((ts, float(row["V"]), float(row["I"]), float(row["W"]), float(row["PF"])))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def known_classes():
    with model_lock:
        clf = models["clf"]
    return sorted(str(c) for c in clf.classes_ if c not in RESERVED)


def _retrain_job():
    try:
        os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        snap = os.path.join(MODEL_BACKUP_DIR, f"training_data_{stamp}.csv")
        with data_lock:
            shutil.copy2(TRAIN_CSV, snap)
            if os.path.isfile(AUTO_CSV):          # automatically collected runs join the snapshot only
                with open(snap, "rb") as fb:
                    data = fb.read()
                needs_nl = bool(data) and not data.endswith((b"\n", b"\r"))
                with open(AUTO_CSV, newline="") as src, open(snap, "a", newline="") as dst:
                    if needs_nl:
                        dst.write("\r\n")
                    next(src, None)               # skip header
                    shutil.copyfileobj(src, dst)
        old_classes = set(known_classes())
        res = train_model.train(snap, "nilm_model.new.pkl", "openset.new.pkl", verbose=False)
        new_classes = set(res["classes"]) - RESERVED

        if res["accuracy"] < RETRAIN_MIN_ACC:
            msg = f"Kept the old model: new accuracy {res['accuracy']:.0%} is below the {RETRAIN_MIN_ACC:.0%} minimum."
            ok = False
        elif not old_classes <= new_classes:
            msg = "Kept the old model: the new one lost a known appliance."
            ok = False
        else:
            for f in (MODEL_FILE, OPENSET_FILE):
                if os.path.isfile(f):
                    shutil.copy2(f, os.path.join(MODEL_BACKUP_DIR, f"{stamp}_{f}"))
            os.replace("nilm_model.new.pkl", MODEL_FILE)
            os.replace("openset.new.pkl", OPENSET_FILE)
            load_models()
            added = sorted(new_classes - old_classes)
            msg = (f"New model is live. Accuracy {res['accuracy']:.0%} on held-out data"
                   + (f"; learned: {', '.join(added)}." if added else "."))
            ok = True
        if res["warnings"]:
            msg += " Note: " + res["warnings"][0]
        retrain_state["last"] = {"ok": ok, "message": msg, "accuracy": round(res["accuracy"], 3),
                                 "classes": sorted(new_classes), "ts": time.time()}
        print("Retrain:", msg)
    except Exception as e:
        retrain_state["last"] = {"ok": False, "message": f"Retraining failed: {type(e).__name__}", "ts": time.time()}
        print("Retrain error:", e)
    finally:
        for f in ("nilm_model.new.pkl", "openset.new.pkl"):
            if os.path.isfile(f):
                os.remove(f)
        retrain_state["running"] = False


# ---------- Web ----------
app = Flask(__name__)

_ana_cache = {}
_last_test = [0.0]
_fails = {}   # ip -> {"n": failed attempts, "first": ts, "locked": until_ts}


def _deny_auth():
    return Response("Sign in required.", 401, {"WWW-Authenticate": 'Basic realm="SmartPlug"'})


def _csrf_ok():
    """State-changing requests must carry a header a cross-site page cannot add."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return True
    return request.headers.get("X-Requested-With") == "SmartPlug"


def _host_name():
    host = request.host or ""
    if host.startswith("["):                 # IPv6 literal like [::1]:8000
        return host.split("]")[0] + "]"
    return host.split(":")[0].lower()


@app.before_request
def guard():
    ip = request.remote_addr or ""
    if not DASH_PASS:
        if ip not in LOCAL_IPS:
            return Response("Remote access is disabled. Set DASHBOARD_PASSWORD and restart the app.", 403)
        # Localhost-only mode still needs protection from web pages open in YOUR browser:
        # the CSRF header blocks cross-site POSTs, the Host check blocks DNS rebinding.
        if _host_name() not in LOCAL_HOSTS:
            return Response("Unexpected Host header.", 403)
        if not _csrf_ok():
            return Response("Missing X-Requested-With header.", 403)
        return None

    now = time.time()
    rec = _fails.get(ip)
    if rec and rec["locked"] > now:
        return Response("Too many failed attempts. Try again in a few minutes.", 429)

    auth = request.authorization
    if auth is None or auth.type != "basic":
        return _deny_auth()
    user_ok = hmac.compare_digest((auth.username or "").encode(), DASH_USER.encode())
    pass_ok = hmac.compare_digest((auth.password or "").encode(), DASH_PASS.encode())
    if not (user_ok and pass_ok):
        rec = _fails.setdefault(ip, {"n": 0, "first": now, "locked": 0.0})
        if now - rec["first"] > 300:
            rec.update(n=0, first=now)
        rec["n"] += 1
        if rec["n"] >= 5:
            rec["locked"] = now + 300
            rec.update(n=0, first=now)
        return _deny_auth()
    _fails.pop(ip, None)

    if not _csrf_ok():
        return Response("Missing X-Requested-With header.", 403)
    return None


@app.after_request
def secure_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.path.startswith("/api") or request.path == "/status":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return app.send_static_file("dashboard.html")


@app.route("/status")
def status():
    now = time.time()
    with state_lock:
        state = dict(latest_state)
        summary = tracker.snapshot(now)
        st = detector.status()
    fresh = state.get("ts") is not None and (now - state["ts"]) < STALE_S
    summary["on_seconds"] = int(now - st["on_since"]) if (st["on_since"] and fresh) else None
    summary["events"] = store.recent(8)
    age = round(now - state["ts"], 1) if state.get("ts") else None
    teach = runs.prompt(now)
    if teach:
        teach["classes"] = known_classes()
    return jsonify({**state, "age": age, "summary": summary, "tariff": TARIFF_R_PER_KWH, "teach": teach})


@app.route("/api/events")
def api_events():
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
    except ValueError:
        limit = 50
    return jsonify(store.recent(limit))


# Named /api/usage on purpose: ad and tracker blockers (uBlock, Brave Shields,
# AdGuard...) block any URL containing "analytics", which made this fail with
# "Failed to fetch" even though the server was fine. The old path still works.
@app.route("/api/usage")
@app.route("/api/analytics")
def api_analytics():
    rng = request.args.get("range", "today")
    if rng not in ("today", "7d", "30d"):
        rng = "today"
    hit = _ana_cache.get(rng)
    if hit and time.time() - hit[0] < 10:
        return jsonify(hit[1])
    try:
        data = analytics.analyze(LOG_FILE, rng, TARIFF_R_PER_KWH, CO2_KG_PER_KWH, UNKNOWN_CONF, MAX_GAP_S)
        chans = channels_configured()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    data["alerts"] = {
        "channels": chans,
        "rules": [describe(r) for r in RULES],
    }
    _ana_cache[rng] = (time.time(), data)
    return jsonify(data)


@app.route("/api/alerts/test", methods=["POST"])
def api_alert_test():
    if time.time() - _last_test[0] < 10:
        return jsonify({"ok": False, "detail": "Wait a few seconds between tests."}), 429
    _last_test[0] = time.time()
    ok, detail = send_alert("SmartPlug test alert: notifications are working.", short="Test alert", wait=False)
    return jsonify({"ok": ok, "detail": detail})


@app.route("/api/health")
def api_health():
    with state_lock:
        ts = latest_state.get("ts")
    with model_lock:
        stats = models["openset"]
    return jsonify({
        "uptime_s": int(time.time() - health["start"]),
        "mqtt": health["mqtt"],
        "last_reading_age": round(time.time() - ts, 1) if ts else None,
        "messages": health["messages"],
        "rejected": health["rejected"],
        "events": store.count(),
        "classes": known_classes(),
        "openset": bool(stats),
        "email": "Email" in channels_configured(),
        "password": bool(DASH_PASS),
        "retrain": {"running": retrain_state["running"], "last": retrain_state["last"]},
    })


@app.route("/api/review")
def api_review():
    since = time.time() - REVIEW_DAYS * 86400
    items = []
    for e in store.review_queue(since, limit=10):
        items.append({
            "id": e["id"], "ts": e["ts"], "label": e["label"], "confidence": e["confidence"],
            "duration_s": e["duration_s"], "watts": e["pre_w"],
        })
    return jsonify({"items": items, "classes": known_classes()})


@app.route("/api/label", methods=["POST"])
def api_label():
    body = request.get_json(silent=True) or {}
    try:
        event_id = int(body.get("event_id"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "detail": "Bad event id."}), 400
    ev = store.get(event_id)
    if not ev or ev["type"] != "OFF" or ev["duration_s"] is None:
        return jsonify({"ok": False, "detail": "That run cannot be labelled."}), 404
    if store.is_reviewed(event_id):
        return jsonify({"ok": False, "detail": "Already handled."}), 409

    if body.get("skip"):
        store.add_review(event_id, "(skipped)", 0)
        return jsonify({"ok": True, "detail": "Skipped."})

    label = str(body.get("label", "")).strip().replace(" ", "_")
    if not LABEL_RE.match(label) or label.lower() in RESERVED:
        return jsonify({"ok": False, "detail": "Use 1-24 letters, digits, - or _ (not 'unknown')."}), 400

    rows = readings_between(ev["ts"] - ev["duration_s"] - 1.5, ev["ts"] + 1.0)[:2000]
    while rows and rows[0][3] < 1.0:      # drop idle readings at the edges of the run
        rows.pop(0)
    while rows and rows[-1][3] < 1.0:
        rows.pop()
    if len(rows) < WINDOW_SIZE + 3:
        return jsonify({"ok": False, "detail": f"Only {len(rows)} readings were logged for this run. Run the appliance for at least a minute so there is enough to learn from."}), 422

    with data_lock:
        new_file = not os.path.isfile(TRAIN_CSV)
        with open(TRAIN_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "label", "V", "I", "W", "PF", "Hz", "kWh"])
            for ts, v, i, watts, pf in rows:
                w.writerow([ts, label, v, i, watts, pf, 50.0, 0.0])   # Hz and kWh are not used by the model
    store.add_review(event_id, label, len(rows))
    return jsonify({"ok": True, "detail": f"Saved {len(rows)} readings as '{label}'. Retrain to learn it.", "rows": len(rows)})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    if not start_retrain():
        return jsonify({"ok": False, "detail": "Already retraining."}), 409
    return jsonify({"ok": True, "detail": "Retraining started."})


@app.route("/api/teach_live", methods=["POST"])
def api_teach_live():
    body = request.get_json(silent=True) or {}
    try:
        run_id = int(body.get("run_id"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "detail": "Bad run id."}), 400
    if body.get("skip"):
        runs.dismiss(run_id)
        return jsonify({"ok": True, "detail": "Skipped."})
    label = str(body.get("label", "")).strip().replace(" ", "_")
    if not LABEL_RE.match(label) or label.lower() in RESERVED:
        return jsonify({"ok": False, "detail": "Use 1-24 letters, digits, - or _ (not 'unknown')."}), 400
    ok, detail, finished = runs.name_run(run_id, label)
    if not ok:
        return jsonify({"ok": False, "detail": detail}), 422
    if finished:
        start_retrain()
        detail += " Learning it now."
    else:
        detail += " I'll keep recording until it stops, then learn it automatically."
    return jsonify({"ok": True, "detail": detail})


@app.route("/api/power")
def api_power():
    rng = request.args.get("range", "today")
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if rng == "30d":
        start, freq, fmt = midnight - timedelta(days=29), "1D", "%d %b"
    elif rng == "7d":
        start, freq, fmt = midnight - timedelta(days=6), "1h", "%a %H:%M"
    else:
        start, freq, fmt = midnight, "10min", "%H:%M"

    empty = jsonify({"labels": [], "values": []})
    try:
        df = pd.read_csv(LOG_FILE, usecols=["time", "W"], on_bad_lines="skip")
    except Exception:
        return empty
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["W"] = pd.to_numeric(df["W"], errors="coerce")
    df = df.dropna()
    df = df[df["time"] >= start]
    if df.empty:
        return empty

    s = df.set_index("time")["W"].resample(freq).mean()
    idx = pd.date_range(start=pd.Timestamp(start).floor(freq), end=pd.Timestamp(now), freq=freq)
    s = s.reindex(idx)
    values = [None if pd.isna(v) else round(float(v), 1) for v in s]
    labels = [t.strftime(fmt) for t in s.index]
    return jsonify({"labels": labels, "values": values})


load_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env_port = os.environ.get("MQTT_PORT", "1883")
    parser.add_argument("--broker", default=os.environ.get("MQTT_BROKER"),
                        help="MQTT broker IP (or set MQTT_BROKER in .env)")
    parser.add_argument("--port", type=int, default=int(env_port) if env_port.isdigit() else 1883,
                        help="MQTT port (or set MQTT_PORT in .env; 8883 for TLS)")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0", help="use 127.0.0.1 to allow this laptop only")
    args = parser.parse_args()
    if not args.broker:
        parser.error("give --broker or set MQTT_BROKER in the .env file")

    os.makedirs(os.path.join(app.static_folder, "renders"), exist_ok=True)
    if not os.path.isfile(os.path.join(app.static_folder, "dashboard.html")):
        raise SystemExit("Missing static\\dashboard.html - create it first.")

    try:
        print("Database snapshot:", store.backup(DB_BACKUP_DIR))
    except Exception as e:
        print("Database snapshot failed:", e)

    replay_log()
    chans = channels_configured()
    print("Alert channels:", ", ".join(chans) if chans else "none configured (alerts still show in the timeline)")
    print("Dashboard:", "password protected" if DASH_PASS else "localhost only (set DASHBOARD_PASSWORD for phone access)")
    with model_lock:
        print("Open-set check:", "on" if models["openset"] else "off")

    threading.Thread(target=start_mqtt, args=(args.broker, args.port), daemon=True).start()
    threading.Thread(target=alert_loop, daemon=True).start()

    app.run(host=args.host, port=args.web_port)