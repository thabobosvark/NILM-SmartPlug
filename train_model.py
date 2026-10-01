"""
Trains the appliance classifier AND the open-set statistics.

Run by hand:   python train_model.py
Also called by the dashboard's "Retrain" button (webapp.py) after you have
labelled new runs, through train().

Outputs: nilm_model.pkl (classifier), openset.pkl (unknown-appliance check).
The accuracy printed comes from a held-out split. The saved classifier is then
refitted on ALL the data, since some classes have very few windows.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import openset
from features import extract_features, WINDOW_SIZE, FEATURE_NAMES

SESSION_GAP_SECONDS = 60  # gap bigger than this = new physical capture run


def assign_sessions(df):
    df = df.sort_values(["label", "timestamp"]).reset_index(drop=True)
    session_ids, counter, prev_label, prev_ts = [], 0, None, None
    for _, row in df.iterrows():
        if row["label"] != prev_label or (prev_ts is not None and row["timestamp"] - prev_ts > SESSION_GAP_SECONDS):
            counter += 1
        session_ids.append(counter)
        prev_label, prev_ts = row["label"], row["timestamp"]
    df["session_id"] = session_ids
    return df


def build_windowed_dataset(df):
    X, y, groups = [], [], []
    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        label = group["label"].iloc[0]
        readings = group[["V", "I", "W", "PF", "Hz", "kWh"]].to_dict("records")
        for i in range(len(readings) - WINDOW_SIZE + 1):
            window = readings[i:i + WINDOW_SIZE]
            X.append(extract_features(window))
            y.append(label)
            groups.append(session_id)
    return np.array(X), np.array(y), np.array(groups)


def fit_openset_from_csv(csv_path="training_data.csv", out_path="openset.pkl"):
    """Build just the unknown-appliance statistics (used if openset.pkl is missing)."""
    df = assign_sessions(pd.read_csv(csv_path))
    X, y, _ = build_windowed_dataset(df)
    stats = openset.fit(X, y)
    joblib.dump(stats, out_path)
    return stats


def train(csv_path="training_data.csv", model_out="nilm_model.pkl", openset_out="openset.pkl", verbose=True):
    log = print if verbose else (lambda *a, **k: None)

    df = pd.read_csv(csv_path)
    log("Labels:")
    log(df["label"].value_counts())
    log()

    df = assign_sessions(df)
    log(f"{df['session_id'].nunique()} distinct capture sessions total.")

    X, y, groups = build_windowed_dataset(df)
    log(f"Windowed dataset: {X.shape[0]} samples, {len(set(y))} classes")
    log()

    sessions_per_label = df.groupby("label")["session_id"].nunique()
    log("Sessions per label:")
    log(sessions_per_label)
    log()

    warnings = []
    if (sessions_per_label < 2).any():
        thin = sessions_per_label[sessions_per_label < 2].index.tolist()
        warnings.append(f"{thin} have only 1 session, so the accuracy below is optimistic.")
        log("WARNING:", warnings[-1])
        log("Using a stratified row-level split (some leakage). Capture a 2nd session for these.\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    clf = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(pd.DataFrame(X_train, columns=FEATURE_NAMES), y_train)
    y_pred = clf.predict(pd.DataFrame(X_test, columns=FEATURE_NAMES))
    acc = float(accuracy_score(y_test, y_pred))

    log(f"Accuracy: {acc:.3f}\n")
    log(classification_report(y_test, y_pred))
    log("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]):
        log(f"  {name:12s} {imp:.3f}")

    # Final classifier uses every window.
    final = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    final.fit(pd.DataFrame(X, columns=FEATURE_NAMES), y)
    joblib.dump(final, model_out)

    stats = openset.fit(X, y)
    joblib.dump(stats, openset_out)
    log(f"\nSaved {model_out} and {openset_out}")
    log("Unknown-appliance limits (distance ratio 1.0 = limit):",
        {c: round(i["thr"], 1) for c, i in stats["classes"].items()})

    return {
        "accuracy": acc,
        "classes": sorted(set(y.tolist())),
        "samples": int(X.shape[0]),
        "warnings": warnings,
    }


if __name__ == "__main__":
    train()
