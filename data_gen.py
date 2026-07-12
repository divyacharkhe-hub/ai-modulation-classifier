"""
data_gen.py
-----------
Generates synthetic I/Q modulation dataset: BPSK, QPSK, 16-QAM, 64-QAM
at multiple SNR levels using an AWGN channel.

Output: data/dataset.npz containing:
    X   -> shape (N, 2, SYMBOLS)   (I and Q channels stacked)
    y   -> shape (N,)              integer labels
    snr -> shape (N,)              SNR (dB) for each sample
    classes -> list of class names in label order
"""

import numpy as np
import os

RNG = np.random.default_rng(42)

MOD_SCHEMES = ["BPSK", "QPSK", "16QAM", "64QAM"]
SNR_RANGE = list(range(-10, 21, 2))   # -10 dB to 20 dB step 2
SYMBOLS_PER_SAMPLE = 128              # length of each I/Q sequence
SAMPLES_PER_CLASS_PER_SNR = 150       # tune this up/down for dataset size


def gray_qam_constellation(order):
    """Build a normalized square-QAM constellation (order = 16 or 64)."""
    m = int(np.sqrt(order))
    levels = np.arange(-(m - 1), m, 2)  # e.g. for m=4 -> [-3,-1,1,3]
    real, imag = np.meshgrid(levels, levels)
    const = (real + 1j * imag).flatten()
    const = const / np.sqrt(np.mean(np.abs(const) ** 2))  # unit average power
    return const


def bpsk_constellation():
    return np.array([-1.0 + 0j, 1.0 + 0j])


def qpsk_constellation():
    const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    return const / np.sqrt(np.mean(np.abs(const) ** 2))


CONSTELLATIONS = {
    "BPSK": bpsk_constellation(),
    "QPSK": qpsk_constellation(),
    "16QAM": gray_qam_constellation(16),
    "64QAM": gray_qam_constellation(64),
}


def generate_symbols(mod, n_symbols):
    const = CONSTELLATIONS[mod]
    idx = RNG.integers(0, len(const), size=n_symbols)
    return const[idx]


def add_awgn(signal, snr_db):
    """Add complex AWGN noise for a target SNR (dB), signal power normalized to 1."""
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = (RNG.normal(0, np.sqrt(noise_power / 2), signal.shape)
              + 1j * RNG.normal(0, np.sqrt(noise_power / 2), signal.shape))
    return signal + noise


def build_dataset():
    X, y, snrs = [], [], []
    for label_idx, mod in enumerate(MOD_SCHEMES):
        for snr_db in SNR_RANGE:
            for _ in range(SAMPLES_PER_CLASS_PER_SNR):
                clean = generate_symbols(mod, SYMBOLS_PER_SAMPLE)
                noisy = add_awgn(clean, snr_db)
                iq = np.stack([noisy.real, noisy.imag], axis=0).astype(np.float32)
                X.append(iq)
                y.append(label_idx)
                snrs.append(snr_db)

    X = np.array(X)
    y = np.array(y)
    snrs = np.array(snrs)

    # shuffle
    perm = RNG.permutation(len(y))
    X, y, snrs = X[perm], y[perm], snrs[perm]
    return X, y, snrs


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    X, y, snrs = build_dataset()
    print("Dataset shape:", X.shape, "labels:", y.shape)
    np.savez_compressed("data/dataset.npz", X=X, y=y, snr=snrs, classes=MOD_SCHEMES)
    print("Saved to data/dataset.npz")
    print("Class distribution:", {m: int(np.sum(y == i)) for i, m in enumerate(MOD_SCHEMES)})
