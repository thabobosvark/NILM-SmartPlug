"""
SQLite store for detected events (smartplug.db).

One connection shared between the MQTT thread (writes) and Flask threads
(reads), guarded by a lock. Plain stdlib, no extra packages.

If you ever add a column, use ALTER TABLE; do not drop the table, or you
lose your event history.
"""
import os
import sqlite3
import threading
import time

COLUMNS = [
    "ts", "type", "label", "confidence", "pre_w", "post_w", "delta_w", "peak_w",
    "pf", "duration_s", "cycles", "note", "prev_label",
]


def _clean(ev):
    out = []
    for c in COLUMNS:
        v = ev.get(c)
        if isinstance(v, float):
            v = round(v, 3)
        out.append(v)
    return out


class EventStore:
    def __init__(self, path="smartplug.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")   # safer if the app is killed mid-write
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts REAL NOT NULL,
                       type TEXT NOT NULL,
                       label TEXT,
                       confidence REAL,
                       pre_w REAL,
                       post_w REAL,
                       delta_w REAL,
                       peak_w REAL,
                       pf REAL,
                       duration_s REAL,
                       cycles INTEGER,
                       note TEXT,
                       prev_label TEXT
                   )"""
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reviews (
                       event_id INTEGER PRIMARY KEY,
                       label TEXT NOT NULL,
                       rows INTEGER,
                       ts REAL
                   )"""
            )
            self._db.commit()

    def insert(self, ev):
        cols = ",".join(COLUMNS)
        marks = ",".join("?" * len(COLUMNS))
        with self._lock:
            cur = self._db.execute(f"INSERT INTO events ({cols}) VALUES ({marks})", _clean(ev))
            self._db.commit()
            return cur.lastrowid

    def update(self, ev):
        if ev.get("id") is None:
            return
        sets = ",".join(f"{c}=?" for c in COLUMNS)
        with self._lock:
            self._db.execute(f"UPDATE events SET {sets} WHERE id=?", _clean(ev) + [ev["id"]])
            self._db.commit()

    def recent(self, limit=20, since=None):
        q = "SELECT * FROM events"
        args = []
        if since is not None:
            q += " WHERE ts >= ?"
            args.append(since)
        q += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def count(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    # ---------- active learning: runs waiting for a human label ----------
    def review_queue(self, since, limit=10, max_conf=0.6, min_duration=20):
        """Finished runs the system was unsure about, least confident first."""
        q = """SELECT * FROM events
               WHERE type='OFF' AND ts >= ? AND duration_s >= ?
                 AND (label IS NULL OR label='unknown' OR confidence < ?)
                 AND id NOT IN (SELECT event_id FROM reviews)
               ORDER BY (confidence IS NOT NULL), confidence ASC, duration_s DESC
               LIMIT ?"""
        with self._lock:
            rows = self._db.execute(q, (since, min_duration, max_conf, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def get(self, event_id):
        with self._lock:
            r = self._db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return dict(r) if r else None

    def is_reviewed(self, event_id):
        with self._lock:
            return self._db.execute("SELECT 1 FROM reviews WHERE event_id=?", (event_id,)).fetchone() is not None

    def add_review(self, event_id, label, rows):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO reviews (event_id, label, rows, ts) VALUES (?,?,?,?)",
                (event_id, label, rows, time.time()),
            )
            self._db.commit()

    # ---------- reliability ----------
    def backup(self, folder="db_backups", keep=7):
        """One snapshot per day using SQLite's online backup. Keeps the newest `keep`."""
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, "smartplug_" + time.strftime("%Y-%m-%d") + ".db")
        if os.path.isfile(dest):
            return dest
        with self._lock:
            out = sqlite3.connect(dest)
            try:
                self._db.backup(out)
            finally:
                out.close()
        old = sorted(f for f in os.listdir(folder) if f.startswith("smartplug_") and f.endswith(".db"))
        for f in old[:-keep]:
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass
        return dest
