"""
train_model.py
---------------
Trains a machine-learning classifier (Random Forest) to identify the
modulation scheme of a signal from its extracted statistical features.

Run:
    python3 train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import FEATURE_NAMES
from signal_generator import MODULATIONS

DATASET_CSV = "dataset.csv"
MODEL_PATH = "modulation_classifier_model.joblib"
SCALER_PATH = "feature_scaler.joblib"


def main():
    df = pd.read_csv(DATASET_CSV)

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["modulation"].astype(str).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nOverall test accuracy: {acc * 100:.2f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=MODULATIONS, target_names=MODULATIONS))

    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=MODULATIONS)
    print(pd.DataFrame(cm, index=MODULATIONS, columns=MODULATIONS))

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
    print(f"Scaler saved -> {SCALER_PATH}")


if __name__ == "__main__":
    main()
