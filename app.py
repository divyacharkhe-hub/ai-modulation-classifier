"""
app.py
------
Streamlit demo: generate a signal at a chosen modulation + SNR,
run it through the trained model, and see the prediction live.
This is the "wow factor" demo piece for portfolio/interviews.

Run:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from signal_generator import MODULATIONS, generate_signal
from features import extract_features, FEATURE_NAMES

MODEL_PATH = "modulation_classifier_model.joblib"
SCALER_PATH = "feature_scaler.joblib"

st.set_page_config(page_title="AI Modulation Classifier", layout="centered")

st.title("📡 AI-Based Automatic Modulation Classification")
st.caption(
    "Generates a digitally modulated signal, extracts statistical features, "
    "and uses a trained ML model to identify the modulation scheme — "
    "an AI extension of classic OFDM / digital modulation signal processing."
)


@st.cache_resource
def load_model():
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


try:
    clf, scaler = load_model()
except FileNotFoundError:
    st.error(
        "Model not found. Run `python3 build_dataset.py` then "
        "`python3 train_model.py` first to train and save the model."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    true_mod = st.selectbox("True modulation (to generate the test signal)", MODULATIONS)
with col2:
    snr_db = st.slider("SNR (dB)", min_value=-10, max_value=25, value=10, step=1)

n_symbols = st.slider("Number of symbols", min_value=100, max_value=2000, value=500, step=100)

if st.button("Generate & Classify", type="primary"):
    sig = generate_signal(true_mod, n_symbols, float(snr_db))
    feats = extract_features(sig).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    pred = clf.predict(feats_scaled)[0]
    proba = clf.predict_proba(feats_scaled)[0]
    proba_df = pd.Series(proba, index=clf.classes_).sort_values(ascending=False)

    st.subheader("Result")
    if pred == true_mod:
        st.success(f"✅ Predicted: **{pred}**  (matches the true modulation)")
    else:
        st.warning(f"⚠️ Predicted: **{pred}**  (true modulation was {true_mod})")

    st.write("Class probabilities:")
    st.bar_chart(proba_df)

    # Constellation diagram
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
    ax1.scatter(sig.real, sig.imag, s=8, alpha=0.5)
    ax1.set_xlabel("In-phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_title(f"Constellation — true: {true_mod}, SNR={snr_db} dB")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_aspect("equal")
    st.pyplot(fig1)

    with st.expander("Extracted feature vector"):
        st.dataframe(pd.Series(feats.flatten(), index=FEATURE_NAMES, name="value"))

st.divider()
st.caption(
    "Model: Random Forest trained on higher-order statistical & cumulant "
    "features (M2/M4/M6, C20/C40/C42, amplitude/phase/frequency statistics) "
    "extracted from simulated BPSK / QPSK / 16-QAM / 64-QAM signals under AWGN."
)
