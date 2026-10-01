# NILM SmartPlug

A low-cost, single-sensor prototype that identifies which appliance is plugged
in and flags when it misbehaves, using nothing more than voltage, current,
and power factor.
The system is built on an ESP32 and a PZEM-004T energy sensor. It detects
ON, OFF, and STEP events on-device using CUSUM, classifies each event,
recognises unknown appliances through open-set recognition, and runs a
per-class Mahalanobis anomaly check on-device. Results are published over
MQTT and rendered on a live web dashboard.
Built in South Africa, with prepaid electricity households in mind.

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [How it works](#how-it-works)
3. [Architecture](#architecture)
4. [Hardware](#hardware)
5. [Software stack](#software-stack)
6. [Repository layout](#repository-layout)
7. [Setup](#setup)
8. [Running the system](#running-the-system)
9. [Dataset](#dataset)
10. [Models and anomaly detection](#models-and-anomaly-detection)
11. [Dashboard](#dashboard)
12. [Alerts](#alerts)
13. [Tests](#tests)
14. [Limitations](#limitations)
15. [Future work](#future-work)
16. [Licence](#licence)

---

## 1. Problem statement

Per-appliance energy monitoring normally requires one smart plug per device.
That is expensive, does not scale to a whole household, and produces a
fragmented user experience.
Non-intrusive load monitoring (NILM) aims to solve this by inferring which
appliances are running from a single point of measurement. Most published
NILM work assumes high sample rates, cloud inference, and closed-set
classification. This project explores the opposite constraints:
- One sensor
- Low sample rate
- Edge-native processing
- Open-set classification
- Anomaly detection on the same signal
The target deployment context is a South African prepaid household where
cost, connectivity, and privacy are all real constraints.
In a prepaid household, the user already knows how much electricity they
have left. The gap is knowing what is eating it. This project addresses that
gap directly.

---

## 2. How it works

The pipeline has five stages.
**Stage 1. Sensing.**
The PZEM-004T reports RMS voltage, RMS current, real power, power factor,
frequency, and accumulated energy. The ESP32 polls it over Modbus RTU at
roughly 8 Hz.  
**Stage 2. Event detection.**
A CUSUM detector runs on the power and power factor streams. It emits ON,
OFF, and STEP events. ON and OFF use threshold plus hysteresis. STEP uses a
two-sided CUSUM with adaptive thresholds based on the current power level.
Heating loads such as the iron use a longer off-hold so thermostat cycles
are not misread as real OFF events.  
**Stage 3. Feature extraction.**
Each event produces a feature vector: delta power, delta power factor, delta
current, pre-event and post-event power and power factor, event duration,
ramp rate, and steady-state variance. Twelve features in total.  
**Stage 4. Classification.**
Events are classified into one of the known appliance classes. The classifier
is an ExtraTrees ensemble trained on hand-captured data. Classification runs
server-side.  
**Stage 5. Open-set and anomaly checks.**
Events that are far from all known class distributions are flagged as
unknown. Events that are close to a known class but statistically abnormal
for that class are flagged as anomalies. The Mahalanobis check runs on the
ESP32. The open-set check runs server-side.
The ESP32 publishes only results over MQTT, never raw waveform data.

---

## 3. Architecture

```

Appliance

|

v

PZEM-004T  --Modbus RTU-->  ESP32

|

+-- CUSUM event detection (on-device)

+-- Feature extraction (on-device)

+-- Mahalanobis anomaly check (on-device)

|

v

MQTT (TLS)

|

v

Backend (Python)

|

+-- Classifier (ExtraTrees)

+-- Open-set recognition

+-- Drift and collective anomaly checks

+-- SQLite storage

+-- Web dashboard

+-- Email and Telegram alerts

```
Classification runs on the server. Event detection and the Mahalanobis
anomaly check run on the ESP32. End-to-end inference-on-chip is future work
and is documented as such.

---

## 4. Hardware

| Component | Notes |
|---|---|
| ESP32 dev board | Any standard ESP32 module |
| PZEM-004T v3.0 | 100 A version, Modbus RTU over UART |
| 5 V supply | Powers ESP32 and PZEM |
| Relay module | Optional, for switching |
| Enclosure | Any mains-safe box |
The bill of materials is under R400 at single-unit pricing. Full breakdown in
the project report.

---

## 5. Software stack

- Python 3.10 or later
- paho-mqtt for MQTT
- pandas and numpy for data handling
- scikit-learn for the classifier, feature importance, and covariance work
- Flask for the dashboard
- python-dotenv for configuration
- joblib for model persistence
- SQLite for local storage

---

## 6. Repository layout

```

NILM-SmartPlug/

├── .env.example           Environment variable template

├── .gitignore

├── README.md

├── requirements.txt

├── static/                Dashboard assets

├── scripts/               All Python source

│   ├── alerts.py          Rule-based alerts and email delivery

│   ├── analytics.py       Per-appliance energy, cost, CO2, standby

│   ├── anomaly.py         Percentile-band anomaly checks

│   ├── collect\_data.py    MQTT capture script for building training data

│   ├── demo\_openset.py    Held-out evaluation of open-set recognition

│   ├── events.py          CUSUM event detector

│   ├── features.py        Shared feature extraction

│   ├── openset.py         Open-set recognition logic

│   ├── store.py           SQLite storage layer

│   ├── tests.py           Unit tests

│   ├── train\_model.py     Model training and session split

│   └── webapp.py          Dashboard and inference server

└── data/

├── training\_data.csv  Labelled training set

├── live\_log.csv       Sample of live operation  

└── images

```

---

## 7. Setup

Clone the repository:

```

git clone git\@github.com:thabobosvark/NILM-SmartPlug.git

cd NILM-SmartPlug

```
Install dependencies:

```

pip install -r requirements.txt

```
Copy the environment template and fill in your values:

```

cp .env.example .env

```
Required environment variables:
| Variable | Purpose |
|---|---|
| SMTP\_USER | Gmail address used to send alerts |
| SMTP\_APP\_PASSWORD | 16 character Google App Password |
| ALERT\_EMAIL\_TO | Comma separated list of recipients |
| SMTP\_SERVER | Defaults to smtp.gmail.com |
| SMTP\_PORT | Defaults to 587 |
| MQTT\_BROKER | Broker IP address |
| MQTT\_PORT | 1883 for plain, 8883 for TLS |
| MQTT\_USER | Broker username |
| MQTT\_PASSWORD | Broker password |
| MQTT\_TLS\_CA | Path to CA certificate if using TLS |
| DASHBOARD\_PASSWORD | Password for the dashboard login |
The** `.env` **file is gitignored. Never commit it.

---

## 8. Running the system

**Step 1. Capture training data.**
With the ESP32 running and the broker reachable, plug in one appliance at a
time and run:

```

cd scripts

python collect\_data.py --label iron --duration 600

```
Repeat for each appliance and each state. The script appends to
** `data/training\_data.csv`**.
**Step 2. Train the model.**

```

python train\_model.py

```
This produces** `nilm\_model.pkl` **and prints per-class metrics using a
session-based split. Sessions are kept whole so windows from the same
recording do not appear in both training and test.
**Step 3. Train the open-set layer.**

```

python demo\_openset.py

```
This holds out one class at a time, fits the open-set model on the
remaining classes, and reports the fraction of held-out windows flagged as
unknown, along with the false alarm rate on known classes.
**Step 4. Start the dashboard.**

```

python webapp.py

```
Open** `http://localhost:5000` **in a browser.

---

## 9. Dataset

The dataset contains labelled captures from the following classes:
- Blender
- Laptop charger
- Lightbulb
- iPhone charger
- iPad charger
- Iron
- Nothing plugged in
Each capture session is recorded separately and labelled at capture time.
Sessions are kept whole when splitting into train and test to prevent
data leakage. An early result of 97 percent accuracy was discarded after
discovering that near-duplicate windows from the same session were appearing
in both train and test. The current reported figures come from a
session-based split.
The iron is the only thermostat-controlled appliance in the set. It cycles
between heating and idle at roughly 20 to 60 second intervals depending on
the dial setting. This is the cycling load used to demonstrate temporal
modeling in the event detector and collective anomaly detection on the
server.
The iPhone and iPad chargers are the hardest pair to separate. They have
nearly identical power draw and power factor. Harmonic features would be
needed to separate them reliably, and this is noted as a limitation.
The dataset is small and comes from a single household. It is included in
the repo as** `data/training\_data.csv`**.

---

## 10. Models and anomaly detection

**Classifier.**
An ExtraTrees ensemble trained on windowed feature vectors. Per-class F1 is
reported rather than overall accuracy, because overall accuracy hides
performance on the minority classes. The confusion matrix is included in
the report.
**Open-set layer.**
Combines a softmax confidence threshold with a Mahalanobis distance from
each class centroid and an autoencoder reconstruction error. The three
signals are combined into a single unknown decision. The threshold is tuned
on held-out classes so unknown detection and false alarm rate can be
reported separately.
**Anomaly layer.**
A two-tier design.
Tier 1 runs on the ESP32. It uses Mahalanobis distance per class on a
reduced feature set. Feature reduction uses the ExtraTrees feature
importances to select the top K continuous features, which keeps the
covariance estimation stable on a small dataset and prevents the distance
calculation from degrading under the curse of dimensionality. The inverse
covariance matrix is computed offline and stored in flash. The on-device
check is a few dozen multiplies.
Tier 2 runs on the server. It covers collective anomaly detection over a
rolling window of events, and EWMA-based drift detection of per-class
feature means over days.
Anomaly thresholds are calibrated to the 99th percentile of normal data.
The reported metric is false positive rate per day, not raw accuracy.
Anomalies for evaluation are synthesised by perturbing normal events by
about 20 percent in power, extending duration by a factor of three, or
cross-feeding one class into another class's model. There are no real
recorded faults in the dataset.

---

## 11. Dashboard

The dashboard is a Flask application served from** `scripts/webapp.py` **with
static assets in** `static/`**. It shows:
- Live power, current, and power factor
- Event timeline with classifications and confidence
- Unknown events with a label picker for the active learning loop
- Per-appliance energy, cost, and CO2 for 1 day, 7 days, and 30 days
- Standby waste summary
- Anomaly log
- Current model version
The dashboard is password protected. Credentials are read from** `.env`**.

---

## 12. Alerts

Alerts are rule-based and configured in** `scripts/alerts.py`**. Each rule
fires once per appliance run so the user is not spammed. Supported rule
types:
-** `max\_on\_minutes`**: appliance has been on for at least N minutes
-** `quiet\_hours`**: appliance is on between two hours of the day
Example rules in the default configuration:
- Iron on for more than 5 minutes
- Iron on between 22:00 and 05:00
- Unknown appliance on for more than 3 minutes
Alerts are delivered by email. Telegram and dashboard toasts are supported
as additional channels.

---

## 13. Tests

Unit tests live in** `scripts/tests.py`**. They cover:
- CUSUM event detection on synthetic ON, OFF, and STEP sequences
- Feature extraction on known windows
- Open-set scoring on held-out classes
- Alert rule triggering and cooldown behaviour
Run with:

```

cd scripts

python tests.py

```

---

## 14. Limitations

- The dataset is small and comes from a single household. Transfer to a
new home is not guaranteed and would require retraining or domain
adaptation.
- The PZEM-004T reports RMS values only. There is no waveform-level data,
which limits harmonic and transient-based discrimination.
- The iPhone and iPad chargers are not reliably separable with the current
feature set.
- Anomalies for evaluation are synthesised by perturbing normal events.
There are no real recorded faults in the dataset.
- Classification runs on the server, not on the ESP32. Only event
detection and the Mahalanobis check run on-device.
- The system has not been tested against adversarial inputs. NILM systems
are a new attack surface and adversarial robustness is not addressed.

---

## 15. Future work

- Waveform-level sampling with a current transformer and a high-speed ADC
for harmonic and transient features. This is the single biggest lever
for accuracy.
- Panel-level disaggregation, moving from one plug to the main breaker.
This is the classic NILM problem and the natural extension of this work.
- Federated learning across multiple households to train a shared model
without moving raw data. Each home trains locally and sends only model
updates. This is the privacy-preserving deployment path.
- A South African appliance dataset, which does not currently exist in
public form.
- On-device classifier inference with TensorFlow Lite Micro. The Mahalanobis
check is already on-device, and this would complete the edge pipeline.
- Formal privacy bounds on the MQTT stream. Right now the privacy story is
a design choice. A differential privacy guarantee would be a real result.
- Concept drift handling across seasons. A model trained in winter may not
hold in summer when the appliance mix changes.

---

## 16. Licence

Released under the MIT Licence. See LICENSE for details.

```
