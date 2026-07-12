# AI-Based Automatic Modulation Classification (AMC)

Deep learning / ML system that takes raw I/Q signal samples and predicts
whether the modulation is **BPSK, QPSK, 16-QAM, or 64-QAM** — across a
range of noisy (AWGN) SNR conditions.

## Why this project
- Direct extension of an OFDM/signal-processing background — pairs well
  with a wireless-communications publication on a resume.
- Pure Python, no hardware required.
- Combines classical DSP/EE knowledge with a modern ML/DL pipeline —
  exactly the "core + AI" combo companies in 5G/6G, defense, and
  research value.

## Project structure
amc_project/
├── data_gen.py          # generates synthetic I/Q dataset (BPSK/QPSK/16QAM/64QAM + AWGN)
├── features.py           # hand-crafted statistical features (moments, cumulants)
├── train_rf.py            # Random Forest baseline (classical ML)
├── train_cnn.py            # 1D-CNN deep learning model (raw I/Q -> class)
├── evaluate.py              # accuracy-vs-SNR curve + confusion matrix
├── streamlit_app.py          # live interactive demo (generate signal, see prediction)
└── data/                       # generated dataset, trained models, plots (created at runtime)
## Setup
```bash
pip install numpy scipy scikit-learn matplotlib joblib tensorflow-cpu streamlit plotly
```

## Run order
```bash
# 1. Generate the dataset (BPSK/QPSK/16QAM/64QAM, SNR -10 to 20 dB)
python3 data_gen.py

# 2. Baseline: classical ML on statistical features
python3 train_rf.py

# 3. Upgrade: deep learning on raw I/Q sequences
python3 train_cnn.py

# 4. Evaluate — generates accuracy-vs-SNR plot + confusion matrix
python3 evaluate.py --model rf
python3 evaluate.py --model cnn

# 5. Live demo for recruiters
streamlit run streamlit_app.py
```

## Results so far
- Random Forest with higher-order moments (M2/M4/M6) + cumulants
  (C20/C21/C40/C41/C42/C60/C63) + classical amplitude/phase/frequency
  stats: **~65% overall accuracy** across all SNRs -10dB to 20dB.
  - BPSK: ~90% precision (very distinguishable — real-valued constellation)
  - QPSK: ~70% precision
  - 16-QAM vs 64-QAM: ~49-50% each — these two are the confusable
    pair. This matches real AMC literature: higher-order QAM schemes
    are genuinely harder to separate, especially at low-to-moderate SNR.
  - `C20_norm` (a cumulant feature) turned out to be the single most
    important feature for the model.
- 1D-CNN (raw I/Q): ~60% in an untuned first pass — more epochs and a
  bigger dataset would push this further.
- Accuracy predictably rises as SNR increases.

## Ideas to extend
1. Add Rayleigh/Rician fading, carrier frequency offset (CFO).
2. Try LSTM or Transformer on raw I/Q instead of CNN.
3. Add more modulation classes: 8-PSK, GFSK, etc.
4. Two-stage classifier to better separate 16-QAM vs 64-QAM.

## How to list this on your resume
> **AI-Based Automatic Modulation Classification for Wireless Signals**
> Built an end-to-end pipeline to classify BPSK/QPSK/16-QAM/64-QAM
> signals from raw I/Q data under AWGN noise; compared a classical ML
> baseline (Random Forest, higher-order cumulant features) against a
> 1D-CNN deep learning model, and visualized accuracy-vs-SNR
> performance trends; deployed a live interactive Streamlit demo.

Pairs naturally with an IJARSCT signal-processing paper — same domain,
shows application of modern AI tools to the same class of problem.
