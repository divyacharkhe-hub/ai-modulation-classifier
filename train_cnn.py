"""
train_cnn.py
------------
Deep learning upgrade: a 1D-CNN that learns directly from raw I/Q
sequences (no hand-crafted features needed). This is the "real AI"
part of the project that recruiters/reviewers care about.

Requires: tensorflow  (pip install tensorflow --break-system-packages)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),          # (2, SYMBOLS) -> channels-first style
        layers.Permute((2, 1)),                    # -> (SYMBOLS, 2) so Conv1D treats I/Q as channels
        layers.Conv1D(64, kernel_size=7, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    data = np.load("data/dataset.npz", allow_pickle=True)
    X, y, snr, classes = data["X"], data["y"], data["snr"], list(data["classes"])

    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, y, snr, test_size=0.25, random_state=42, stratify=y
    )

    model = build_model(input_shape=X.shape[1:], num_classes=len(classes))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

    model.save("data/model_cnn.keras")
    np.savez("data/cnn_test_split.npz", X_test=X_test, y_test=y_test, snr_test=snr_test)
    print("Model saved to data/model_cnn.keras")


if __name__ == "__main__":
    main()
