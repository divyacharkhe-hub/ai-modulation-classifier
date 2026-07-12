"""
features.py
------------
Extracts classic Automatic Modulation Classification (AMC) features
from complex I/Q signal samples. These are the same style of
higher-order statistical / cumulant features used in real AMC
research papers, making this project technically credible (not a toy).
"""

import numpy as np
from scipy.stats import kurtosis

FEATURE_NAMES = [
    "mean_amplitude", "std_amplitude", "kurtosis_amplitude",
    "mean_phase", "std_phase", "std_inst_freq",
    "M2", "M4", "M6",
    "C20", "C40", "C42",
    "papr", "zero_crossing_rate",
]


def extract_features(iq_samples: np.ndarray) -> np.ndarray:
    """Compute a fixed-length feature vector from complex I/Q samples."""
    x = np.asarray(iq_samples)
    amplitude = np.abs(x)
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase)

    mean_amp = np.mean(amplitude)
    std_amp = np.std(amplitude)
    kurt_amp = kurtosis(amplitude, fisher=False)

    mean_phase = np.mean(phase)
    std_phase = np.std(phase)
    std_freq = np.std(inst_freq) if len(inst_freq) > 0 else 0.0

    m2 = np.abs(np.mean(x ** 2))
    m4 = np.abs(np.mean(x ** 4))
    m6 = np.abs(np.mean(x ** 6))

    m2_full = np.mean(np.abs(x) ** 2)
    m4_full = np.mean(np.abs(x) ** 4)
    c20 = np.abs(np.mean(x ** 2))
    c40 = np.abs(np.mean(x ** 4) - 3 * np.mean(x ** 2) ** 2)
    c42 = np.abs(m4_full - np.abs(np.mean(x ** 2)) ** 2 - 2 * m2_full ** 2)

    power = amplitude ** 2
    papr = np.max(power) / np.mean(power) if np.mean(power) > 0 else 0.0

    real_part = np.real(x)
    zero_crossings = np.sum(np.diff(np.sign(real_part)) != 0)
    zcr = zero_crossings / len(real_part)

    return np.array([
        mean_amp, std_amp, kurt_amp,
        mean_phase, std_phase, std_freq,
        m2, m4, m6,
        c20, c40, c42,
        papr, zcr,
    ])


if __name__ == "__main__":
    from signal_generator import generate_signal
    sig = generate_signal("16QAM", n_symbols=500, snr_db=15, seed=1)
    feats = extract_features(sig)
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"{name:22s}: {val:.4f}")
