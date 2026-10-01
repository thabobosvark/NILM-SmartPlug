"""
Rule-based alerts and email delivery.

Edit RULES to change what triggers an alert. Each rule fires ONCE per run of
an appliance (a run = from ON until OFF), so you will not get spammed.

Rule types:
  max_on_minutes   appliance has been on for at least `minutes`
  quiet_hours      appliance is on between start_hour and end_hour (local time)

"label" limits a rule to one appliance (iron, laptop, blender, lightbulb,
iPhone, iPad). Leave it out to apply the rule to any appliance.

Email is sent exactly like the Lab 4 SMTP practical: connect to the Gmail SMTP
server on port 587, upgrade with STARTTLS, log in with an App Password, send.
Settings are read from environment variables, which are loaded from the .env
file in this folder (never put the password in code):
  SMTP_USER           the Gmail address that sends the alerts
  SMTP_APP_PASSWORD   the 16-character Google App Password (not your normal password)
  ALERT_EMAIL_TO      who receives alerts; several addresses separated by commas
Optional: SMTP_SERVER (default smtp.gmail.com), SMTP_PORT (default 587).

Test it from the terminal, with the full SMTP conversation printed like the lab:
  python alerts.py
"""
import smtplib
import ssl
import threading
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.utils import formatdate
import os
from pathlib import Path

# Load .env from this file's folder, so it works no matter where you run Python from.
# Variables already set in the terminal win over the file (override=False).
# This runs when alerts is imported, which webapp.py does BEFORE it reads
# DASHBOARD_PASSWORD / MQTT_* - so the whole app sees the .env values.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass   # python-dotenv not installed: fall back to real environment variables only

RULES = [
    {"id": "iron_long", "type": "max_on_minutes", "label": "iron", "minutes": 5},
    {"id": "iron_late", "type": "quiet_hours", "label": "iron", "start_hour": 22, "end_hour": 5, "grace_s": 60},
    {"id": "unknown_long", "type": "max_on_minutes", "label": "unknown", "minutes": 3},
]

NAMES = {
    "iron": "Iron", "laptop": "Laptop", "blender": "Blender", "lightbulb": "Lightbulb",
    "iPhone": "iPhone charger", "iPad": "iPad charger", "unknown": "Unknown appliance",
}

SMTP_TIMEOUT_S = 15        # never let a dead mail server hang the alert thread
_mail_lock = threading.Lock()


def _name(label):
    return NAMES.get(label, "An appliance")


def describe(rule):
    who = _name(rule.get("label")) if rule.get("label") else "Any appliance"
    if rule["type"] == "max_on_minutes":
        return f"{who} on for more than {rule['minutes']} min"
    if rule["type"] == "quiet_hours":
        return f"{who} on between {rule['start_hour']:02d}:00 and {rule['end_hour']:02d}:00"
    return rule.get("id", "rule")


# ---------- Email channel (Lab 4 SMTP, made reusable) ----------
def email_configured():
    return bool(
        os.environ.get("SMTP_USER")
        and os.environ.get("SMTP_APP_PASSWORD")
        and os.environ.get("ALERT_EMAIL_TO")
    )


def channels_configured():
    return ["Email"] if email_configured() else []


def _err(e):
    """Short error text that never includes credentials."""
    return type(e).__name__


def send_email(subject, body, debug=False):
    """Returns (ok, detail). debug=True prints the whole SMTP conversation, as in the lab."""
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_APP_PASSWORD", "").replace(" ", "")   # spaces in the App Password do not matter
    to = [a.strip() for a in os.environ.get("ALERT_EMAIL_TO", "").split(",") if a.strip()]
    if not (user and password and to):
        return False, "Email not configured"
    server_name = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        port = 587

    # Build the message: MIMEText sets Content-Type text/plain; utf-8 so any symbol survives
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject.replace("\r", " ").replace("\n", " ")[:120]
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg["Date"] = formatdate(localtime=True)

    try:
        with _mail_lock:                                   # one email at a time
            # Step 1: open the SMTP connection (port 587 starts unencrypted)
            with smtplib.SMTP(server_name, port, timeout=SMTP_TIMEOUT_S) as server:
                if debug:
                    server.set_debuglevel(1)               # print the SMTP conversation
                # Step 2: upgrade to TLS. The default context also checks the server's certificate.
                server.starttls(context=ssl.create_default_context())
                # Step 3: authenticate with the Gmail address and the App Password
                server.login(user, password)
                # Step 4: send. The block's exit sends QUIT and closes the connection.
                server.sendmail(user, to, msg.as_string())
        return True, "sent"
    except smtplib.SMTPAuthenticationError:
        return False, "login rejected (check SMTP_USER and the App Password)"
    except Exception as e:
        return False, _err(e)


def send_alert(text, short=None, wait=True):
    """Email the alert. Returns (ok, detail). `wait` is kept so callers do not change."""
    if not email_configured():
        return False, "Email not configured"
    subject = "SmartPlug alert: " + (short or text)
    body = f"{text}\n\nTime: {datetime.now():%Y-%m-%d %H:%M:%S}\nSent by your SmartPlug monitor."
    ok, detail = send_email(subject, body)
    return ok, "Email: " + ("sent" if ok else detail)


# ---------- Rules ----------
class AlertEngine:
    def __init__(self, rules):
        self.rules = rules
        self.fired = {}   # (rule_id, run_start_ts) -> time fired

    def check(self, now, status, fresh):
        """status = detector.status(); returns a list of alert dicts to deliver."""
        out = []
        if not fresh or status["state"] != "on" or status["on_since"] is None:
            return out

        label = status["label"]
        on_s = now - status["on_since"]
        hour = datetime.fromtimestamp(now).hour

        for r in self.rules:
            if r.get("label") and r["label"] != label:
                continue
            key = (r["id"], status["on_since"])
            if key in self.fired:
                continue

            msg = short = None
            if r["type"] == "max_on_minutes" and on_s >= r["minutes"] * 60:
                mins = int(on_s // 60)
                msg = f"{_name(label)} has been on for {mins} min."
                short = f"{_name(label)} on {mins} min"
            elif r["type"] == "quiet_hours" and on_s >= r.get("grace_s", 60):
                s, e = r["start_hour"], r["end_hour"]
                inside = (hour >= s or hour < e) if s > e else (s <= hour < e)
                if inside:
                    msg = f"{_name(label)} is on during quiet hours ({s:02d}:00 to {e:02d}:00)."
                    short = f"{_name(label)} on at night"

            if msg:
                self.fired[key] = now
                out.append({"rule": r["id"], "label": label, "message": msg, "short": short, "ts": now})

        for k in [k for k, t in self.fired.items() if now - t > 86400]:
            del self.fired[k]
        return out


if __name__ == "__main__":
    # Same idea as the lab: send one message and show the SMTP conversation.
    if not email_configured():
        raise SystemExit("Set SMTP_USER, SMTP_APP_PASSWORD and ALERT_EMAIL_TO in the .env file first.")
    ok, detail = send_email("SmartPlug test email", "If you can read this, SmartPlug email alerts work.", debug=True)
    print("Email sent successfully!" if ok else f"An error occurred: {detail}")