"""
signal_generator.py
--------------------
Generates synthetic digitally modulated signals (BPSK, QPSK, 16-QAM, 64-QAM),
passes them through an AWGN channel at a chosen SNR, and returns the
complex baseband I/Q samples.

This builds on the OFDM/digital-modulation groundwork already covered
in the author's IJARSCT paper, extending it toward an AI-based
Automatic Modulation Classification (AMC) system.
"""

import numpy as np

MODULATIONS = ["BPSK", "QPSK", "16QAM", "64QAM"]


def _normalize_power(symbols: np.ndarray) -> np.ndarray:
    """Scale symbols to unit average power."""
    power = np.mean(np.abs(symbols) ** 2)
    return symbols / np.sqrt(power)


def generate_symbols(modulation: str, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random complex symbols for the given modulation scheme."""
    if modulation == "BPSK":
        bits = rng.integers(0, 2, n_symbols)
        symbols = 2 * bits - 1  # -1, +1
        symbols = symbols.astype(complex)

    elif modulation == "QPSK":
        bits = rng.integers(0, 2, (n_symbols, 2))
        i = 2 * bits[:, 0] - 1
        q = 2 * bits[:, 1] - 1
        symbols = (i + 1j * q) / np.sqrt(2)

    elif modulation == "16QAM":
        levels = np.array([-3, -1, 1, 3])
        i = rng.choice(levels, n_symbols)
        q = rng.choice(levels, n_symbols)
        symbols = i + 1j * q

    elif modulation == "64QAM":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        i = rng.choice(levels, n_symbols)
        q = rng.choice(levels, n_symbols)
        symbols = i + 1j * q

    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return _normalize_power(symbols)


def add_awgn(symbols: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN noise to achieve the target SNR (dB)."""
    signal_power = np.mean(np.abs(symbols) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape)
    )
    return symbols + noise


def generate_signal(modulation: str, n_symbols: int, snr_db: float, seed: int = None) -> np.ndarray:
    """Generate a single noisy modulated signal (I/Q samples)."""
    rng = np.random.default_rng(seed)
    clean = generate_symbols(modulation, n_symbols, rng)
    noisy = add_awgn(clean, snr_db, rng)
    return noisy


if __name__ == "__main__":
    # quick sanity check
    sig = generate_signal("QPSK", n_symbols=10, snr_db=10, seed=1)
    print("Sample QPSK signal (SNR=10dB):")
    print(sig)
