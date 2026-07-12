"""
train_rf.py
-----------
Beginner-friendly baseline: Random Forest on hand-crafted statistical
features (no deep learning needed). Good first milestone before CNN/LSTM.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from features import extract_features_batch


def main():
    data = np.load("data/dataset.npz", allow_pickle=True)
    X, y, snr, classes = data["X"], data["y"], data["snr"], list(data["classes"])

    print("Extracting features...")
    feats = extract_features_batch(X)

    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        feats, y, snr, test_size=0.25, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nOverall test accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=classes))

    joblib.dump(clf, "data/rf_model.joblib")
    np.savez("data/rf_test_split.npz", X_test=X_test, y_test=y_test, snr_test=snr_test)
    print("Model saved to data/rf_model.joblib")


if __name__ == "__main__":
    main()
