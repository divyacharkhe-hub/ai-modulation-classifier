"""
features.py
------------
Converts raw I/Q sequences into hand-crafted statistical features that a
classical ML model (Random Forest / SVM) can use. These features are the
kind used in real AMC papers: amplitude, phase, and higher-order moments.
"""

import numpy as np


def extract_features(iq_sample):
    """
    iq_sample: shape (2, N) -> [I, Q]
    returns: 1D feature vector
    """
    I, Q = iq_sample[0], iq_sample[1]
    complex_sig = I + 1j * Q
    amp = np.abs(complex_sig)
    phase = np.unwrap(np.angle(complex_sig))

    amp_norm = amp / (np.mean(amp) + 1e-9)

    feats = [
        np.mean(amp), np.std(amp), np.var(amp),
        np.mean(amp_norm ** 2), np.mean(amp_norm ** 4),   # 2nd/4th moments
        np.std(phase), np.mean(np.abs(phase)),
        np.mean(I), np.std(I), np.mean(Q), np.std(Q),
        np.max(amp) - np.min(amp),
        np.mean(I ** 2 + Q ** 2),                          # avg power
        kurtosis(amp), kurtosis(I), kurtosis(Q),
        skewness(I), skewness(Q),
    ]
    return np.array(feats, dtype=np.float32)


def kurtosis(x):
    x = x - np.mean(x)
    return np.mean(x ** 4) / (np.mean(x ** 2) ** 2 + 1e-9)


def skewness(x):
    x = x - np.mean(x)
    return np.mean(x ** 3) / (np.mean(x ** 2) ** 1.5 + 1e-9)


def extract_features_batch(X):
    """X: shape (N, 2, samples) -> returns (N, num_features)"""
    return np.array([extract_features(x) for x in X])
