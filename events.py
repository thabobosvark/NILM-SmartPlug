"""
Event detection for the SmartPlug power stream.

Emits three kinds of event from a stream of (timestamp, watts) readings:
  ON    an appliance started drawing power      (threshold + hysteresis)
  OFF   the load dropped back to idle           (threshold + hold time)
  STEP  the power level changed while running   (two-sided CUSUM)

Pure Python, no MQTT or database code, so it can be replayed offline
(see replay_events.py) and unit tested.

Usage:
    det = EventDetector()
    emitted, updated = det.update(ts, w, pf, label, conf)

  emitted  = events created by this reading (insert them into the store)
  updated  = events whose numbers or label were finalised by this reading
             (the same dict objects, so write them back to the store)
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from statistics import median

# Classifier labels that carry no information about WHICH appliance is running
SKIP_VOTE = {None, "warming_up", "nothing_plugged"}


@dataclass
class EventConfig:
    # --- ON / OFF (hysteresis) ---
    on_w: float = 3.0              # reading must reach this to start an ON
    off_w: float = 1.5             # absolute OFF threshold
    off_rel: float = 0.05          # OFF threshold is also 5% of the running level
    on_confirm: int = 2            # consecutive readings needed for ON
    off_hold_s: float = 14.0       # time below the OFF threshold before OFF
    heating_w: float = 1000.0      # loads at or above this use the longer hold
    heating_off_hold_s: float = 60.0   # thermostat cycles (iron) are not OFF
    # --- STEP (two-sided CUSUM) ---
    settle_n: int = 3              # readings used to settle a new level
    k_abs: float = 2.0             # CUSUM allowance: max(k_abs, k_rel * level)
    k_rel: float = 0.05
    h_abs: float = 8.0             # CUSUM threshold: max(h_abs, h_rel * level)
    h_rel: float = 0.15
    # --- labelling by classifier votes ---
    vote_min_since: int = 5        # only count votes once the classifier window is inside the new segment
    votes_needed: int = 3
    max_label_wait: int = 12       # readings to wait before giving up on a label
    # --- housekeeping ---
    gap_reset_s: float = 60.0      # silence longer than this resets the detector
    pre_n: int = 3                 # idle readings kept to estimate the pre-event level


class EventDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or EventConfig()
        self.reset(resumed=True)

    def reset(self, resumed=False):
        self.state = "idle"
        self.idle_w = deque(maxlen=self.cfg.pre_n)
        self.on_buf = []
        self.seg = None
        self.last_ts = None
        self.resumed = resumed   # True = we may have missed the real start of a run

    # ------------------------------------------------------------------
    def update(self, ts, w, pf=0.0, label=None, conf=0.0):
        emitted, updated = [], []
        if self.last_ts is not None and ts - self.last_ts > self.cfg.gap_reset_s:
            self.reset(resumed=(self.state == "on"))
        self.last_ts = ts

        if self.state == "idle":
            active = self._idle(ts, w, pf, emitted)
        else:
            active = self._on(ts, w, pf, emitted, updated)

        if active and self.state == "on":
            self._vote(label, conf, updated)
        return emitted, updated

    def status(self):
        if self.state != "on" or self.seg is None:
            return {"state": "idle", "on_since": None, "label": None, "cycles": 0}
        s = self.seg
        return {"state": "on", "on_since": s["on_ts"], "label": s["label"], "cycles": s["cycles"]}

    # ------------------------------------------------------------------
    @staticmethod
    def _make(typ, ts, **kw):
        ev = {
            "id": None, "ts": ts, "type": typ, "label": None, "confidence": None,
            "pre_w": None, "post_w": None, "delta_w": None, "peak_w": None, "pf": None,
            "duration_s": None, "cycles": None, "note": None, "prev_label": None,
        }
        ev.update(kw)
        if ev["delta_w"] is None and ev["pre_w"] is not None and ev["post_w"] is not None:
            ev["delta_w"] = ev["post_w"] - ev["pre_w"]
        return ev

    @staticmethod
    def _majority(s):
        if not s["votes"]:
            return None, None
        lab, n = s["votes"].most_common(1)[0]
        return lab, round(s["conf"][lab] / n, 3)

    # ------------------------------------------------------------------
    def _idle(self, ts, w, pf, emitted):
        if w >= self.cfg.on_w:
            self.on_buf.append((ts, w, pf))
            if len(self.on_buf) >= self.cfg.on_confirm:
                self._start_on(emitted)
                return True
        else:
            self.on_buf = []
            self.resumed = False
            self.idle_w.append(w)
        return False

    def _start_on(self, emitted):
        cfg = self.cfg
        buf = self.on_buf
        ws = [x[1] for x in buf]
        pre = median(self.idle_w) if self.idle_w else 0.0
        level = median(ws)
        ev = self._make(
            "ON", buf[0][0], pre_w=pre, post_w=level, peak_w=max(ws), pf=buf[-1][2],
            note="Already running when monitoring resumed" if self.resumed else None,
        )
        self.seg = {
            "on_ts": buf[0][0], "level": level, "readings": deque(ws, maxlen=30),
            "peak": max(ws), "g_pos": 0.0, "g_neg": 0.0,
            "settle_left": cfg.settle_n, "settle_ev": ev,
            "since": len(ws), "votes": Counter(), "conf": defaultdict(float),
            "pending": ev, "wait": 0, "label": None, "label_conf": None,
            "cycles": 0, "off_since": None,
        }
        self.state = "on"
        self.on_buf = []
        self.resumed = False
        emitted.append(ev)

    # ------------------------------------------------------------------
    def _on(self, ts, w, pf, emitted, updated):
        cfg, s = self.cfg, self.seg
        off_thr = max(cfg.off_w, cfg.off_rel * s["level"])

        if w < off_thr:
            if s["off_since"] is None:
                s["off_since"] = ts
            hold = cfg.heating_off_hold_s if s["level"] >= cfg.heating_w else cfg.off_hold_s
            if ts - s["off_since"] >= hold:
                self._end_on(emitted, updated)
            return False

        if s["off_since"] is not None:      # power came back: a thermostat cycle, not an OFF
            s["cycles"] += 1
            s["off_since"] = None
            s["g_pos"] = 0.0
            s["g_neg"] = 0.0

        s["since"] += 1
        s["readings"].append(w)
        s["peak"] = max(s["peak"], w)

        if s["settle_left"] > 0:            # let a new level settle before arming CUSUM
            s["settle_left"] -= 1
            s["level"] = median(list(s["readings"])[-3:])
            if s["settle_left"] == 0 and s["settle_ev"] is not None:
                ev = s["settle_ev"]
                ev["post_w"] = s["level"]
                ev["delta_w"] = ev["post_w"] - ev["pre_w"]
                ev["peak_w"] = s["peak"]
                ev["pf"] = pf
                updated.append(ev)
                s["settle_ev"] = None
            return True

        lvl = s["level"]
        k = max(cfg.k_abs, cfg.k_rel * lvl)
        h = max(cfg.h_abs, cfg.h_rel * lvl)
        d = w - lvl
        s["g_pos"] = max(0.0, s["g_pos"] + d - k)
        s["g_neg"] = max(0.0, s["g_neg"] - d - k)
        if s["g_pos"] > h or s["g_neg"] > h:
            self._step(ts, w, pf, emitted, updated)
        return True

    def _step(self, ts, w, pf, emitted, updated):
        s = self.seg
        if s["pending"] is not None:
            self._finalize(updated)
        ev = self._make("STEP", ts, pre_w=s["level"], post_w=w, peak_w=w, pf=pf)
        ev["prev_label"] = s["label"]
        s.update(
            g_pos=0.0, g_neg=0.0, settle_left=self.cfg.settle_n, settle_ev=ev,
            since=0, pending=ev, wait=0, votes=Counter(), conf=defaultdict(float), peak=w,
        )
        emitted.append(ev)

    def _end_on(self, emitted, updated):
        s = self.seg
        if s["pending"] is not None:
            self._finalize(updated)
        lab, conf = s["label"], s["label_conf"]
        if lab is None:
            lab, conf = self._majority(s)
        ev = self._make(
            "OFF", s["off_since"], pre_w=s["level"], post_w=0.0,
            label=lab, confidence=conf,
            duration_s=s["off_since"] - s["on_ts"], cycles=s["cycles"],
        )
        emitted.append(ev)
        self.state = "idle"
        self.seg = None
        self.idle_w.clear()

    # ------------------------------------------------------------------
    def _vote(self, label, conf, updated):
        s = self.seg
        if s is None:
            return
        if s["since"] >= self.cfg.vote_min_since and label not in SKIP_VOTE:
            s["votes"][label] += 1
            s["conf"][label] += conf
        if s["pending"] is not None:
            s["wait"] += 1
            if sum(s["votes"].values()) >= self.cfg.votes_needed or s["wait"] >= self.cfg.max_label_wait:
                self._finalize(updated)

    def _finalize(self, updated):
        s = self.seg
        ev = s["pending"]
        lab, conf = self._majority(s)
        ev["label"], ev["confidence"] = lab, conf
        if ev["type"] == "STEP" and lab and s["label"] and lab != s["label"]:
            s["on_ts"] = ev["ts"]           # a different appliance: restart the run timer
        if lab:
            s["label"], s["label_conf"] = lab, conf
        s["pending"] = None
        updated.append(ev)