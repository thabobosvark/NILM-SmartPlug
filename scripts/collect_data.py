"""
Data capture script (MQTT version) for building your real training set.

HOW TO USE:
1. Make sure your ESP32 is running the MQTT firmware and Mosquitto is running.
2. Plug in ONE appliance in ONE state (e.g. laptop charger, actively charging).
3. Run:  python collect_data.py --label laptop_active --duration 2400
   (the broker address, port and login come from the .env file; you can still
   override the address with --broker 192.168.10.241)
4. Let it run the full duration without touching the appliance.
   Ctrl+C stops early; every reading is already saved.
5. Unplug / change state, run again with a different --label.
6. Repeat for every appliance x state combo.

Every run APPENDS to the same CSV (default: training_data.csv), so over
multiple sessions you build one file with all your labeled data.
Wait at least 60 s between two runs of the same label so they count as
separate sessions.

Settings read from .env (same names the dashboard uses):
  MQTT_BROKER, MQTT_PORT (1883 plain, 8883 TLS), MQTT_USER, MQTT_PASSWORD,
  MQTT_TLS_CA (full path to ca.crt)
"""
import argparse
import csv
import os
import time
import json
from pathlib import Path

import paho.mqtt.client as mqtt

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass   # python-dotenv not installed: real environment variables only

FIELDNAMES = ["timestamp", "label", "V", "I", "W", "PF", "Hz", "kWh"]


def main():
    env_port = os.environ.get("MQTT_PORT", "1883")
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=os.environ.get("MQTT_BROKER"),
                        help="MQTT broker IP (default: MQTT_BROKER from .env)")
    parser.add_argument("--port", type=int, default=int(env_port) if env_port.isdigit() else 1883,
                        help="MQTT broker port (default: MQTT_PORT from .env, else 1883)")
    parser.add_argument("--topic", default="nilm/plug1/reading", help="Topic the ESP32 publishes readings to")
    parser.add_argument("--device-id", default=None,
                         help="Only log readings from this device_id (default: log all)")
    parser.add_argument("--label", required=True, help="e.g. laptop_active")
    parser.add_argument("--duration", type=int, default=120, help="seconds to capture (default 120)")
    parser.add_argument("--out", default="training_data.csv", help="CSV file to append to")
    args = parser.parse_args()
    if not args.broker:
        parser.error("give --broker or set MQTT_BROKER in the .env file")

    file_exists = os.path.isfile(args.out)
    csv_file = open(args.out, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    count = 0
    start_time = None  # set once the first reading arrives, not at connect time

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"Connected to MQTT broker. Subscribing to '{args.topic}'...")
            client.subscribe(args.topic)
        else:
            print(f"Failed to connect, return code {rc} "
                  "(check MQTT_USER, MQTT_PASSWORD, MQTT_TLS_CA and the port)")

    def on_message(client, userdata, msg):
        nonlocal count, start_time
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  skipped malformed message: {msg.payload}")
            return
        if not isinstance(payload, dict):
            return

        if args.device_id and payload.get("device_id") != args.device_id:
            return

        if start_time is None:
            start_time = time.time()
            print(f"First reading received - capturing '{args.label}' for {args.duration}s...")

        row = {
            "timestamp": time.time(),
            "label": args.label,
            "V": payload.get("V", 0.0),
            "I": payload.get("I", 0.0),
            "W": payload.get("W", 0.0),
            "PF": payload.get("PF", 0.0),
            "Hz": payload.get("Hz", 0.0),
            "kWh": payload.get("kWh", 0.0),
        }
        writer.writerow(row)
        csv_file.flush()
        count += 1
        print(f"  [{count}] W={row['W']:.1f} PF={row['PF']:.2f} I={row['I']:.3f}")

        if time.time() - start_time >= args.duration:
            client.disconnect()

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

    user = os.environ.get("MQTT_USER")
    if user:
        client.username_pw_set(user, os.environ.get("MQTT_PASSWORD", ""))
    ca = os.environ.get("MQTT_TLS_CA")
    if ca:
        client.tls_set(ca_certs=ca)

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to broker {args.broker}:{args.port}"
          f"{' (TLS)' if ca else ''}{' as ' + user if user else ''}...")
    client.connect(args.broker, args.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped early by user.")
        client.disconnect()

    csv_file.close()
    print(f"\nDone. {count} readings appended to {args.out} with label '{args.label}'.")


if __name__ == "__main__":
    main()