# NILM SmartPlug

A low-cost prototype that identifies which appliance is plugged in — and flags when it misbehaves — from a single voltage/current/power-factor sensor.

Built on an ESP32 and PZEM-004T. The system detects ON/OFF/STEP events on-device with CUSUM, classifies each event server-side, recognises unknown appliances via open-set recognition, and runs a per-class Mahalanobis anomaly check on-device.

Runs end-to-end: sensor → ESP32 → MQTT → classifier → live web dashboard.

Built in South Africa, with prepaid electricity households in mind.
