# AI-Based Automatic Modulation Classification (AMC)

A software-only, Python-based project that uses machine learning to
automatically identify the digital modulation scheme (BPSK, QPSK,
16-QAM, or 64-QAM) of a wireless signal — extending classic
OFDM / digital modulation signal-processing work with an AI layer.

This project is designed as a natural extension of an existing
OFDM communication system simulation and IJARSCT publication: same
signal-processing foundation, now paired with a trained ML classifier
and an interactive demo.

## Why this project

- **Pure software** — no hardware/SDR required, runs entirely in Python.
- **AI + Telecom combo** — Automatic Modulation Classification is an
  active research area used in 5G/6G, cognitive radio, spectrum
  monitoring, and defense/SIGINT applications.
- **Technically credible features** — uses the same higher-order
  statistical moments and cumulants (M2, M4, M6, C20, C40, C42) found
  in published AMC research, not arbitrary made-up features.

## How it works

1. **Signal generation** (`signal_generator.py`) — generates BPSK,
   QPSK, 16-QAM, and 64-QAM symbols, then passes them through an AWGN
   channel at a chosen SNR (signal-to-noise ratio).
2. **Feature extraction** (`features.py`) — computes 14 statistical
   features per signal: amplitude/phase/frequency statistics, raw
   moments (M2, M4, M6), and cumulants (C20, C40, C42).
3. **Dataset generation** (`build_dataset.py`) — builds a labeled
   dataset across all 4 modulations and SNR levels from -5 dB to 20 dB.
4. **Model training** (`train_model.py`) — trains a Random Forest
   classifier on the extracted features and reports accuracy,
   classification report, confusion matrix, and feature importances.
5. **Evaluation** (`evaluate.py`) — generates the two standard AMC
   benchmark plots: accuracy vs SNR, and a confusion matrix heatmap.
6. **Live demo** (`app.py`) — a Streamlit web app where you pick a
   modulation + SNR, generate a fresh signal, and watch the model
   classify it in real time, complete with a constellation diagram.

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
# 1. Build the labeled dataset
python3 build_dataset.py

# 2. Train the model
python3 train_model.py

# 3. Generate evaluation plots
python3 evaluate.py

# 4. Launch the interactive demo
streamlit run app.py
```

## Results (on the included dataset)

- Overall test accuracy: **~81%** on held-out data across all SNR
  levels (including very noisy -5 dB cases, which are intentionally
  hard).
- Overall accuracy across the **full dataset** (including training
  data): ~95%.
- BPSK is classified almost perfectly (as expected — it's the
  simplest scheme). 16-QAM and 64-QAM are sometimes confused with
  each other at low SNR, which matches real-world AMC research
  findings.
- See `accuracy_vs_snr.png`, `accuracy_vs_snr_per_modulation.png`,
  and `confusion_matrix.png` after running `evaluate.py`.

## Possible extensions

- Add more modulation schemes (8-PSK, 256-QAM).
- Try a 1D-CNN directly on raw I/Q samples instead of hand-crafted
  features (deep learning approach).
- Add Rayleigh/Rician fading channel models in addition to AWGN.
- Deploy the Streamlit app publicly (Streamlit Community Cloud) and
  link it in your portfolio/resume.

## Resume / LinkedIn description

> **AI-Based Automatic Modulation Classification System** — Built an
> end-to-end ML pipeline in Python that classifies wireless signal
> modulation schemes (BPSK/QPSK/16-QAM/64-QAM) from higher-order
> statistical features, achieving ~95% accuracy across SNR levels
> from -5 to 20 dB. Extended prior OFDM/digital-modulation signal
> simulation work with a Random Forest classifier and an interactive
> Streamlit demo.

## Project structure

```
modulation_classifier/
├── signal_generator.py    # generates modulated signals + AWGN noise
├── features.py             # extracts statistical/cumulant features
├── build_dataset.py         # builds labeled dataset -> dataset.csv
├── train_model.py           # trains & saves the Random Forest model
├── evaluate.py               # generates accuracy/confusion-matrix plots
├── app.py                     # Streamlit live demo
├── requirements.txt
└── README.md
```
