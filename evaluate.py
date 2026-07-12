"""
evaluate.py
-----------
Generates two key plots recruiters/reviewers always want to see in an
AMC project:
  1. Accuracy vs SNR curve
  2. Confusion matrix (at high SNR, where the model should do its best)

Works with the Random Forest baseline by default. If a CNN model
(model_cnn.keras) exists, pass --model cnn to evaluate that instead.
"""

import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from features import extract_features_batch


def load_rf_predictions():
    clf = joblib.load("data/rf_model.joblib")
    split = np.load("data/rf_test_split.npz")
    X_test, y_test, snr_test = split["X_test"], split["y_test"], split["snr_test"]
    preds = clf.predict(X_test)
    return y_test, preds, snr_test


def load_cnn_predictions():
    import tensorflow as tf
    model = tf.keras.models.load_model("data/model_cnn.keras")
    split = np.load("data/cnn_test_split.npz")
    X_test, y_test, snr_test = split["X_test"], split["y_test"], split["snr_test"]
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return y_test, preds, snr_test


def main(model_type):
    data = np.load("data/dataset.npz", allow_pickle=True)
    classes = list(data["classes"])

    if model_type == "rf":
        y_test, preds, snr_test = load_rf_predictions()
    else:
        y_test, preds, snr_test = load_cnn_predictions()

    # --- Accuracy vs SNR ---
    snrs_unique = sorted(np.unique(snr_test))
    accs = []
    for s in snrs_unique:
        mask = snr_test == s
        accs.append(np.mean(preds[mask] == y_test[mask]))

    plt.figure(figsize=(7, 5))
    plt.plot(snrs_unique, accs, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Classification Accuracy")
    plt.title(f"Accuracy vs SNR ({model_type.upper()} model)")
    plt.grid(True)
    plt.savefig(f"data/accuracy_vs_snr_{model_type}.png", dpi=150, bbox_inches="tight")
    print(f"Saved data/accuracy_vs_snr_{model_type}.png")

    # --- Confusion matrix at high SNR (>=10 dB) ---
    high_snr_mask = snr_test >= 10
    cm = confusion_matrix(y_test[high_snr_mask], preds[high_snr_mask])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix at SNR >= 10dB ({model_type.upper()})")
    plt.savefig(f"data/confusion_matrix_{model_type}.png", dpi=150, bbox_inches="tight")
    print(f"Saved data/confusion_matrix_{model_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "cnn"], default="rf")
    args = parser.parse_args()
    main(args.model)
