"""
build_dataset.py
-----------------
Builds a labeled dataset of feature vectors across all modulation
schemes and a range of SNR levels, then saves it as a CSV.

Run:
    python3 build_dataset.py
"""

import numpy as np
import pandas as pd

from signal_generator import MODULATIONS, generate_signal
from features import extract_features, FEATURE_NAMES

# ---- Config ----
N_SYMBOLS = 500          # symbols per signal instance
SNR_RANGE_DB = np.arange(-5, 21, 2)   # -5 dB to 20 dB in steps of 2
SAMPLES_PER_CLASS_PER_SNR = 60        # signal instances per (modulation, SNR) pair
OUTPUT_CSV = "dataset.csv"
SEED = 42


def build_dataset():
    rng_master = np.random.default_rng(SEED)
    rows = []

    for modulation in MODULATIONS:
        for snr_db in SNR_RANGE_DB:
            for _ in range(SAMPLES_PER_CLASS_PER_SNR):
                seed = int(rng_master.integers(0, 1_000_000))
                sig = generate_signal(modulation, N_SYMBOLS, float(snr_db), seed=seed)
                feats = extract_features(sig)
                row = list(feats) + [snr_db, modulation]
                rows.append(row)

    columns = FEATURE_NAMES + ["snr_db", "modulation"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset built: {len(df)} rows -> {OUTPUT_CSV}")
    print(df["modulation"].value_counts())
    return df


if __name__ == "__main__":
    build_dataset()
