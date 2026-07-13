"""
streamlit_app.py
-----------------
Live demo styled as a signal-analyzer console: pick a modulation + SNR,
watch the constellation sweep onto the "scope" like a real spectrum
analyzer building up a trace, then watch the classifier lock on.

Run with:  streamlit run streamlit_app.py
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib

from data_gen import generate_symbols, add_awgn, MOD_SCHEMES, SYMBOLS_PER_SAMPLE
from features import extract_features, extract_features_batch, FEATURE_NAMES

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@st.cache_resource(show_spinner="Training Random Forest model (first run only, ~20s)...")
def get_rf_model():
    """Train the RF baseline on the fly and cache it — no need to ship a
    pre-trained joblib file in the repo."""
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    rng = np.random.default_rng(7)
    X_list, y_list = [], []
    for label_idx, mod in enumerate(MOD_SCHEMES):
        for snr_db in range(-10, 21, 2):
            for _ in range(60):  # smaller than the full dataset script, but enough for a live demo
                clean = generate_symbols(mod, SYMBOLS_PER_SAMPLE)
                noisy = add_awgn(clean, snr_db)
                iq = np.stack([noisy.real, noisy.imag], axis=0).astype(np.float32)
                X_list.append(extract_features(iq))
                y_list.append(label_idx)

    X = np.array(X_list)
    y = np.array(y_list)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                  random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

st.set_page_config(page_title="AMC Console", layout="wide", page_icon="📡")

# ------------------------------------------------------------------
# THEME — dark signal-analyzer console: phosphor green trace on navy,
# monospace instrument readout type, LED-segment confidence meters.
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-void: #0a0e17;
    --panel: #10182a;
    --grid: #1f2b42;
    --signal: #35f2a0;
    --signal-dim: #1c7a56;
    --amber: #ffb020;
    --red: #ff5c5c;
    --text: #e6edf3;
    --text-dim: #7c8797;
}

.stApp {
    background-color: var(--bg-void);
    background-image:
        linear-gradient(rgba(53,242,160,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(53,242,160,0.06) 1px, transparent 1px);
    background-size: 32px 32px;
    animation: gridDrift 14s linear infinite;
    color: var(--text);
}
@keyframes gridDrift {
    0%   { background-position: 0 0, 0 0; }
    100% { background-position: 64px 64px, 64px 64px; }
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* console header */
.console-header {
    border: 1px solid var(--grid);
    background: linear-gradient(180deg, var(--panel), var(--bg-void));
    border-radius: 6px;
    padding: 18px 24px;
    margin-bottom: 18px;
}
.console-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--signal);
    letter-spacing: 0.06em;
    animation: titlePulse 2.6s ease-in-out infinite;
    margin: 0;
}
@keyframes titlePulse {
    0%, 100% { text-shadow: 0 0 10px rgba(53,242,160,0.30); }
    50%      { text-shadow: 0 0 22px rgba(53,242,160,0.65); }
}
.console-sub { color: var(--text-dim); font-size: 0.92rem; margin-top: 4px; }

/* SWEEP button — pulsing glow so it invites a click */
div.stButton > button {
    background: linear-gradient(180deg, #12351f, #0d2417) !important;
    color: var(--signal) !important;
    border: 1px solid var(--signal-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.08em;
    animation: btnGlow 2s ease-in-out infinite;
}
div.stButton > button:hover {
    border-color: var(--signal) !important;
    transform: translateY(-1px);
}
@keyframes btnGlow {
    0%, 100% { box-shadow: 0 0 6px rgba(53,242,160,0.25); }
    50%      { box-shadow: 0 0 18px rgba(53,242,160,0.55); }
}

/* readout panels */
.panel {
    border: 1px solid var(--grid);
    background: var(--panel);
    border-radius: 6px;
    padding: 16px 18px;
}
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--grid);
    padding-bottom: 6px;
    margin-bottom: 10px;
}

/* decode readout */
.decode-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--signal);
    text-shadow: 0 0 16px rgba(53,242,160,0.4);
    animation: lockOn 0.4s ease-out;
}
@keyframes lockOn {
    0% { opacity: 0; transform: scale(0.9); filter: blur(4px); }
    100% { opacity: 1; transform: scale(1); filter: blur(0); }
}

/* LED bar meter */
.led-row { display: flex; align-items: center; margin: 7px 0; gap: 10px; }
.led-label { width: 62px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-dim); }
.led-track {
    flex: 1; height: 14px; border-radius: 3px; background: #060a12;
    border: 1px solid var(--grid); overflow: hidden; position: relative;
}
.led-fill {
    height: 100%;
    background: repeating-linear-gradient(90deg, var(--signal) 0px, var(--signal) 6px, #060a12 6px, #060a12 8px);
    transition: width 0.6s ease-out;
}
.led-pct { width: 46px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-dim); text-align: right; }

/* cumulant readout table */
.stat-row { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; padding: 3px 0; border-bottom: 1px dashed var(--grid); }
.stat-key { color: var(--text-dim); }
.stat-val { color: var(--signal); }

/* radar sweep overlay — rotating scan line over the constellation */
.radar-wrap { position: relative; }
.radar-sweep {
    position: absolute; inset: 0; pointer-events: none; z-index: 5;
    background: conic-gradient(from 0deg, rgba(53,242,160,0.22), transparent 35%);
    animation: radarSpin 2.6s linear infinite;
    border-radius: 4px;
    mix-blend-mode: screen;
}
@keyframes radarSpin { 100% { transform: rotate(360deg); } }

/* pulse ring on lock-on */
.lock-ring {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: var(--signal); margin-right: 8px; position: relative; top: 1px;
    box-shadow: 0 0 0 0 rgba(53,242,160,0.6);
    animation: ringPulse 1.4s ease-out infinite;
}
@keyframes ringPulse {
    0%   { box-shadow: 0 0 0 0 rgba(53,242,160,0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(53,242,160,0); }
    100% { box-shadow: 0 0 0 0 rgba(53,242,160,0); }
}
.lock-ring.mismatch { background: var(--amber); box-shadow: 0 0 0 0 rgba(255,176,32,0.55); animation: ringPulseAmber 1.4s ease-out infinite; }
@keyframes ringPulseAmber {
    0%   { box-shadow: 0 0 0 0 rgba(255,176,32,0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(255,176,32,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,176,32,0); }
}

div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: var(--signal); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="console-header">
        <p class="console-title">📡 AMC-01 &nbsp; SIGNAL ANALYZER CONSOLE</p>
        <p class="console-sub">Automatic Modulation Classification &middot;
        cumulant &amp; higher-order-statistics engine &middot; live AWGN simulation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# CONTROLS
# ------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 1])
with c1:
    true_mod = st.selectbox("MODULATION (ground truth)", MOD_SCHEMES)
with c2:
    snr_db = st.slider("SNR (dB)", -10, 25, 10, step=1)
with c3:
    n_symbols = st.select_slider("SYMBOLS", options=[100, 200, 500, 1000, 2000], value=500)
with c4:
    model_options = ["Random Forest"] + (["1D-CNN"] if TF_AVAILABLE else [])
    model_choice = st.radio("MODEL", model_options, label_visibility="visible")
    if not TF_AVAILABLE:
        st.caption("⚠ TensorFlow not installed — CNN model unavailable, Random Forest only.")

run = st.button("▶  SWEEP & CLASSIFY", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
scope_col, readout_col = st.columns([1.5, 1])

# ------------------------------------------------------------------
# MAIN SEQUENCE
# ------------------------------------------------------------------
if run:
    clean = generate_symbols(true_mod, n_symbols)
    noisy = add_awgn(clean, snr_db)
    iq = np.stack([noisy.real, noisy.imag], axis=0).astype(np.float32)

    with scope_col:
        st.markdown(
            '<div class="panel radar-wrap"><div class="panel-label">CONSTELLATION SCOPE — live sweep</div>'
            '<div class="radar-sweep"></div>',
            unsafe_allow_html=True,
        )
        scope_ph = st.empty()

        # animate the trace building up point-by-point, like a real scope
        # accumulating persistence across sweeps
        n_frames = 18
        chunk = max(1, len(noisy) // n_frames)
        for f in range(1, n_frames + 1):
            shown = noisy[: f * chunk]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=shown.real, y=shown.imag, mode="markers",
                marker=dict(size=6, color="#35f2a0", opacity=0.75,
                            line=dict(width=0)),
            ))
            fig.update_layout(
                paper_bgcolor="#10182a", plot_bgcolor="#0a0e17",
                xaxis=dict(range=[-2.2, 2.2], gridcolor="#1f2b42", zerolinecolor="#1f2b42", title="I"),
                yaxis=dict(range=[-2.2, 2.2], gridcolor="#1f2b42", zerolinecolor="#1f2b42", title="Q", scaleanchor="x"),
                margin=dict(l=10, r=10, t=10, b=10), height=430,
                font=dict(family="JetBrains Mono", color="#7c8797"),
                showlegend=False,
            )
            scope_ph.plotly_chart(fig, use_container_width=True, key=f"scope_{f}")
            time.sleep(0.03)
        st.markdown("</div>", unsafe_allow_html=True)

    with readout_col:
        st.markdown('<div class="panel"><div class="panel-label">HIGHER-ORDER STATISTICS</div>', unsafe_allow_html=True)
        feats = extract_features(iq)
        # show a compact, most-informative subset of the full feature vector
        show_names = ["M2", "M4", "M6", "C20_norm", "C40_norm", "C42_norm", "C63_norm"]
        rows = ""
        for name in show_names:
            val = feats[FEATURE_NAMES.index(name)]
            rows += f'<div class="stat-row"><span class="stat-key">{name}</span><span class="stat-val">{val:.4f}</span></div>'
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown("</div><br>", unsafe_allow_html=True)

        decode_ph = st.empty()
        decode_ph.markdown(
            '<div class="panel"><div class="panel-label">DECODE</div>'
            '<span class="mono" style="color:#7c8797;">scanning...</span></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.5)

        if model_choice == "Random Forest":
            clf = get_rf_model()
            probs = clf.predict_proba(feats.reshape(1, -1))[0]
        else:
            model = tf.keras.models.load_model("data/model_cnn.keras")
            probs = model.predict(iq.reshape(1, 2, -1), verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_mod = MOD_SCHEMES[pred_idx]
        correct = pred_mod == true_mod

        led_rows = ""
        order = np.argsort(probs)[::-1]
        for i in order:
            pct = probs[i] * 100
            led_rows += (
                '<div class="led-row">'
                f'<span class="led-label mono">{MOD_SCHEMES[i]}</span>'
                f'<div class="led-track"><div class="led-fill" style="width:{pct}%;"></div></div>'
                f'<span class="led-pct mono">{pct:.1f}%</span>'
                '</div>'
            )

        status_color = "var(--signal)" if correct else "var(--amber)"
        status_text = "MATCH" if correct else "MISMATCH vs ground truth"
        ring_class = "lock-ring" if correct else "lock-ring mismatch"

        # typewriter reveal of the predicted modulation name — fun to watch,
        # and doubles as a natural pause before the "lock-on" moment
        for k in range(len(pred_mod) + 1):
            partial = pred_mod[:k] + ("▌" if k < len(pred_mod) else "")
            decode_ph.markdown(
                f'''
                <div class="panel">
                    <div class="panel-label">DECODE</div>
                    <div class="decode-value">{partial}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            time.sleep(0.08)

        decode_ph.markdown(
            f'''
            <div class="panel">
                <div class="panel-label">DECODE</div>
                <div class="decode-value">{pred_mod}</div>
                <div class="mono" style="color:{status_color}; font-size:0.8rem; margin:6px 0 14px;">
                    <span class="{ring_class}"></span>{status_text} &nbsp;|&nbsp; input: {true_mod} @ {snr_db} dB
                </div>
                {led_rows}
            </div>
            ''',
            unsafe_allow_html=True,
        )
else:
    with scope_col:
        st.markdown(
            '<div class="panel" style="height:470px; display:flex; align-items:center; justify-content:center;">'
            '<span class="mono" style="color:#7c8797;">◎ awaiting sweep — press START to generate a signal</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with readout_col:
        st.markdown(
            '<div class="panel" style="height:470px; display:flex; align-items:center; justify-content:center; text-align:center; padding:24px;">'
            '<span class="mono" style="color:#7c8797; font-size:0.85rem;">Random Forest model uses M2/M4/M6 moments + '
            'C20/C21/C40/C41/C42/C60/C63 cumulants extracted from the I/Q signal.<br><br>'
            '1D-CNN model reads the raw I/Q sequence directly, no hand-crafted features.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="mono" style="color:#7c8797; font-size:0.78rem; margin-top:18px;">'
    'TIP — sweep at -8 dB then again at 18 dB and compare the LED confidence bars: '
    'this is the same accuracy-vs-SNR trend real 5G/6G and defense spectrum-sensing systems are benchmarked on.'
    '</p>',
    unsafe_allow_html=True,
)
