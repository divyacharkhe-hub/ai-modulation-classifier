"""
features.py
------------
Higher-order statistics feature extraction for Automatic Modulation
Classification — this is the classical (pre-deep-learning) approach used
in real AMC research (Swami & Sadler, "Hierarchical digital modulation
classification using cumulants", IEEE Trans. Commun., 2000).

Two families of features:
  1. Raw moments: M2, M4, M6  (2nd/4th/6th order moments of |x|)
  2. Cumulants:    C20, C21, C40, C41, C42, C60, C63
     Cumulants remove the Gaussian-noise contribution that pure moments
     don't, which is why they classify modulation better than simple
     statistics at low-to-moderate SNR.
Plus standard amplitude/phase/frequency statistics used in classical
signal-processing based classifiers.
"""

import numpy as np


def _moments(x):
    """x: complex 1D array (zero-mean assumed after removing DC)."""
    x2 = x ** 2
    xc = np.conj(x)
    absx2 = np.abs(x) ** 2

    M20 = np.mean(x2)
    M21 = np.mean(x * xc)              # = E[|x|^2] = average power
    M40 = np.mean(x2 ** 2)
    M41 = np.mean(x2 * x * xc)          # E[x^3 x*]
    M42 = np.mean(absx2 ** 2)           # E[|x|^4]
    M60 = np.mean(x2 ** 3)
    M63 = np.mean(absx2 ** 3)           # E[|x|^6]
    return M20, M21, M40, M41, M42, M60, M63


def _cumulants(x):
    M20, M21, M40, M41, M42, M60, M63 = _moments(x)

    C20 = M20
    C21 = M21
    C40 = M40 - 3 * M20 ** 2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    C60 = M60 - 15 * M20 * M40 + 30 * M20 ** 3
    C63 = M63 - 9 * C21 * C42 - 6 * C21 ** 3   # valid for circularly-symmetric constellations
    return C20, C21, C40, C41, C42, C60, C63


def kurtosis(v):
    v = v - np.mean(v)
    return np.mean(v ** 4) / (np.mean(v ** 2) ** 2 + 1e-9)


def skewness(v):
    v = v - np.mean(v)
    return np.mean(v ** 3) / (np.mean(v ** 2) ** 1.5 + 1e-9)


def extract_features(iq_sample):
    """
    iq_sample: shape (2, N) -> [I, Q]
    returns: 1D feature vector combining classical stats + moments + cumulants
    """
    I, Q = iq_sample[0], iq_sample[1]
    x = I + 1j * Q
    x = x - np.mean(x)  # remove any DC offset before higher-order stats

    amp = np.abs(x)
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase)  # crude instantaneous-frequency proxy

    amp_norm = amp / (np.mean(amp) + 1e-9)

    # --- classical amplitude/phase/frequency statistics ---
    classical = [
        np.mean(amp), np.std(amp), np.var(amp),
        np.mean(amp_norm ** 2), np.mean(amp_norm ** 4),
        np.std(phase), np.mean(np.abs(phase)),
        np.std(inst_freq), np.mean(np.abs(inst_freq)),
        np.max(amp) - np.min(amp),
        kurtosis(amp), kurtosis(I), kurtosis(Q),
        skewness(I), skewness(Q),
    ]

    # --- raw moments M2 / M4 / M6 (of |x|) ---
    M2 = np.mean(amp ** 2)
    M4 = np.mean(amp ** 4)
    M6 = np.mean(amp ** 6)
    moments_feats = [M2, M4, M6]

    # --- cumulants C20 / C21 / C40 / C41 / C42 / C60 / C63 ---
    C20, C21, C40, C41, C42, C60, C63 = _cumulants(x)
    # normalize by signal power (C21) so features are SNR/scale-robust
    power = np.abs(C21) + 1e-9
    cumulant_feats = [
        np.abs(C20) / power,
        np.abs(C40) / (power ** 2),
        np.abs(C41) / (power ** 2),
        np.abs(C42) / (power ** 2),
        np.abs(C60) / (power ** 3),
        np.abs(C63) / (power ** 3),
    ]

    return np.array(classical + moments_feats + cumulant_feats, dtype=np.float32)


def extract_features_batch(X):
    """X: shape (N, 2, samples) -> returns (N, num_features)"""
    return np.array([extract_features(x) for x in X])


FEATURE_NAMES = [
    "amp_mean", "amp_std", "amp_var", "amp_norm2", "amp_norm4",
    "phase_std", "phase_meanabs", "freq_std", "freq_meanabs",
    "amp_range", "kurt_amp", "kurt_I", "kurt_Q", "skew_I", "skew_Q",
    "M2", "M4", "M6",
    "C20_norm", "C40_norm", "C41_norm", "C42_norm", "C60_norm", "C63_norm",
]
    
