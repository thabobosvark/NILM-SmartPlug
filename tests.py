"""
Automated checks. Run from the project folder:   python tests.py

Uses a temporary database and does not touch smartplug.db, live_log.csv or
training_data.csv. Needs nilm_model.pkl and training_data.csv to exist.
"""
import base64
import os
import tempfile
import unittest
import warnings

warnings.filterwarnings("ignore")
_TMP = tempfile.mkdtemp()
os.environ["SMARTPLUG_DB"] = os.path.join(_TMP, "test.db")

import numpy as np
import pandas as pd

import anomaly
import openset
from alerts import AlertEngine
from events import EventDetector
from store import EventStore
from train_model import assign_sessions, build_windowed_dataset


def _basic(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


class OpenSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = assign_sessions(pd.read_csv("training_data.csv"))
        cls.X, cls.y, _ = build_windowed_dataset(df)

    def test_known_windows_pass(self):
        stats = openset.fit(self.X, self.y)
        sel = self.X[self.y == "laptop"]
        flagged = np.mean([openset.score(stats, x)[0] > 1.0 for x in sel])
        self.assertLess(flagged, 0.02)

    def test_unseen_blender_is_flagged(self):
        keep = (self.y != "blender") & (self.y != "nothing_plugged")
        stats = openset.fit(self.X[keep], self.y[keep])
        flagged = np.mean([openset.score(stats, x)[0] > 1.0 for x in self.X[self.y == "blender"]])
        self.assertGreater(flagged, 0.9)

    def test_empty_stats_never_flag(self):
        self.assertEqual(openset.score(None, [0.0] * 12)[0], 0.0)


class AnomalyTests(unittest.TestCase):
    def setUp(self):
        self.det = anomaly.AnomalyDetector(anomaly.build_profiles("training_data.csv"))

    def feed(self, label, w, n, v=245.0, hz=50.0, conf=0.95):
        out = []
        for i in range(n):
            out += self.det.update(1000.0 + i * 7, label, conf, {"V": v, "Hz": hz, "W": w})
        return out

    def test_normal_laptop_is_quiet(self):
        self.assertEqual(self.feed("laptop", 40.0, 10), [])

    def test_laptop_power_surge(self):
        self.assertEqual([a["kind"] for a in self.feed("laptop", 500.0, 3)], ["level"])

    def test_single_spike_ignored(self):
        self.assertEqual(self.feed("laptop", 500.0, 2), [])

    def test_low_confidence_ignored(self):
        self.assertEqual(self.feed("laptop", 500.0, 5, conf=0.4), [])

    def test_overvoltage(self):
        self.assertEqual([a["kind"] for a in self.feed(None, 0.0, 3, v=270.0)], ["supply"])

    def test_cooldown_limits_repeats(self):
        self.assertEqual(len(self.feed("laptop", 500.0, 40)), 1)


class AlertRuleTests(unittest.TestCase):
    RULES = [{"id": "r", "type": "max_on_minutes", "label": "iron", "minutes": 5}]

    def status(self, label="iron", since=0.0):
        return {"state": "on", "on_since": since, "label": label, "cycles": 0}

    def test_fires_once_after_threshold(self):
        eng = AlertEngine(self.RULES)
        self.assertEqual(eng.check(200.0, self.status(), True), [])
        self.assertEqual(len(eng.check(301.0, self.status(), True)), 1)
        self.assertEqual(eng.check(400.0, self.status(), True), [])

    def test_other_appliance_ignored(self):
        self.assertEqual(AlertEngine(self.RULES).check(400.0, self.status("laptop"), True), [])

    def test_stale_data_never_alerts(self):
        self.assertEqual(AlertEngine(self.RULES).check(400.0, self.status(), False), [])

class EmailTests(unittest.TestCase):
    """Uses a fake SMTP server, so nothing is really sent."""

    ENV = {"SMTP_USER": "me@gmail.com", "SMTP_APP_PASSWORD": "abcd efgh ijkl mnop",
           "ALERT_EMAIL_TO": "a@icloud.com, b@mycput.ac.za"}

    def setUp(self):
        import alerts
        self.alerts = alerts
        self.calls = []
        calls = self.calls

        class FakeSMTP:
            fail_login = False

            def __init__(s, host, port, timeout=None):
                calls.append(("connect", host, port))

            def __enter__(s): return s
            def __exit__(s, *a): calls.append(("quit",))
            def set_debuglevel(s, n): calls.append(("debug", n))
            def starttls(s, context=None): calls.append(("starttls", context is not None))
            def login(s, u, p):
                if FakeSMTP.fail_login:
                    import smtplib
                    raise smtplib.SMTPAuthenticationError(535, b"bad")
                calls.append(("login", u, p))
            def sendmail(s, frm, to, text): calls.append(("sendmail", frm, to, text))

        self.Fake = FakeSMTP
        self._orig = alerts.smtplib.SMTP
        alerts.smtplib.SMTP = FakeSMTP
        self._env = {k: os.environ.get(k) for k in self.ENV}
        os.environ.update(self.ENV)

    def tearDown(self):
        self.alerts.smtplib.SMTP = self._orig
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_follows_the_lab_steps_in_order(self):
        ok, detail = self.alerts.send_alert("Iron has been on for 5 min.", short="Iron on 5 min")
        self.assertTrue(ok, detail)
        self.assertEqual([c[0] for c in self.calls], ["connect", "starttls", "login", "sendmail", "quit"])
        self.assertEqual(self.calls[0][1:], ("smtp.gmail.com", 587))
        self.assertTrue(self.calls[1][1])                      # TLS used a verifying context
        self.assertEqual(self.calls[2][2], "abcdefghijklmnop")  # spaces stripped from the App Password

    def test_recipients_and_subject(self):
        self.alerts.send_alert("Iron has been on for 5 min.", short="Iron on 5 min")
        _, frm, to, text = [c for c in self.calls if c[0] == "sendmail"][0]
        self.assertEqual(frm, "me@gmail.com")
        self.assertEqual(to, ["a@icloud.com", "b@mycput.ac.za"])
        self.assertIn("Subject: SmartPlug alert: Iron on 5 min", text)

    def test_header_injection_blocked(self):
        self.alerts.send_email("Hi\r\nBcc: evil@x.com", "body")
        text = [c for c in self.calls if c[0] == "sendmail"][0][3]
        self.assertNotIn("\nBcc:", text)

    def test_bad_login_reports_cleanly(self):
        self.Fake.fail_login = True
        ok, detail = self.alerts.send_alert("x")
        self.assertFalse(ok)
        self.assertIn("login rejected", detail)
        self.assertNotIn("abcd", detail)

    def test_not_configured(self):
        os.environ.pop("ALERT_EMAIL_TO")
        self.assertEqual(self.alerts.channels_configured(), [])
        self.assertFalse(self.alerts.send_alert("x")[0])


class EventTests(unittest.TestCase):
    def test_on_then_off(self):
        det, evs = EventDetector(), []
        for i in range(8):
            em, up = det.update(i * 7.0, 60.0, 0.5, "laptop", 0.95)
            evs += em
        for i in range(8, 14):
            em, up = det.update(i * 7.0, 0.0, 0.0, "nothing_plugged", 1.0)
            evs += em
        self.assertEqual([e["type"] for e in evs], ["ON", "OFF"])


class StoreTests(unittest.TestCase):
    def test_review_queue_and_backup(self):
        st = EventStore(os.path.join(_TMP, "s.db"))
        a = st.insert({"ts": 1000.0, "type": "OFF", "label": "unknown", "confidence": 0.2, "duration_s": 90.0})
        b = st.insert({"ts": 2000.0, "type": "OFF", "label": "laptop", "confidence": 0.99, "duration_s": 90.0})
        self.assertEqual([e["id"] for e in st.review_queue(0)], [a])
        st.add_review(a, "kettle", 12)
        self.assertEqual(st.review_queue(0), [])
        self.assertTrue(b)
        self.assertTrue(os.path.isfile(st.backup(os.path.join(_TMP, "bk"))))


class WebSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import webapp
        cls.w = webapp
        cls.c = webapp.app.test_client()

    def setUp(self):
        self.w.DASH_PASS = ""
        self.w._fails.clear()

    def test_remote_blocked_without_password(self):
        r = self.c.get("/status", environ_overrides={"REMOTE_ADDR": "192.168.1.50"})
        self.assertEqual(r.status_code, 403)

    def test_localhost_allowed_without_password(self):
        self.assertEqual(self.c.get("/status").status_code, 200)

    def test_password_required_and_lockout(self):
        self.w.DASH_PASS = "pw"
        ip = {"REMOTE_ADDR": "10.1.1.1"}
        self.assertEqual(self.c.get("/status", environ_overrides=ip).status_code, 401)
        codes = [self.c.get("/status", headers=_basic("admin", "no"), environ_overrides=ip).status_code for _ in range(6)]
        self.assertEqual(codes[-1], 429)
        ok = self.c.get("/status", headers=_basic("admin", "pw"), environ_overrides={"REMOTE_ADDR": "10.1.1.2"})
        self.assertEqual(ok.status_code, 200)

    def test_post_needs_csrf_header(self):
        self.w.DASH_PASS = "pw"
        ip = {"REMOTE_ADDR": "10.1.1.3"}
        r = self.c.post("/api/alerts/test", headers=_basic("admin", "pw"), environ_overrides=ip)
        self.assertEqual(r.status_code, 403)

    def test_bad_sensor_values_rejected(self):
        good = dict(V=245.0, I=1.0, W=50.0, PF=0.5, Hz=50.0, kWh=0.0)
        self.assertTrue(self.w.valid_reading(good))
        self.assertFalse(self.w.valid_reading({**good, "V": float("nan")}))
        self.assertFalse(self.w.valid_reading({**good, "W": 99999.0}))
        self.assertFalse(self.w.valid_reading({**good, "PF": 3.0}))

    def test_label_validation(self):
        r = self.c.post("/api/label", json={"event_id": 1, "label": "<b>x</b>"},
                        headers={"X-Requested-With": "SmartPlug"})
        self.assertIn(r.status_code, (400, 404))


if __name__ == "__main__":
    unittest.main(verbosity=1)
