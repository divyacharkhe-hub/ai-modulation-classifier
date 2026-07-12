# OFDM Communication System Simulation

A Python simulation of an end-to-end **OFDM (Orthogonal Frequency Division Multiplexing)** transceiver — the core waveform technology behind 4G LTE and 5G NR. Built from scratch using NumPy (no telecom libraries), demonstrating a full understanding of the physical layer signal chain.

## What it does

The simulation implements the complete chain:

```
Random Bits → 16-QAM Modulation → IFFT (OFDM) → Cyclic Prefix
→ Wireless Channel (AWGN / Rayleigh Fading) → Remove CP → FFT
→ 16-QAM Demodulation → BER Calculation
```

It then evaluates system performance across a range of SNR values and produces:

- **BER vs SNR curve** comparing AWGN and Rayleigh fading channels
- **Constellation diagram** showing received symbol scatter at a given SNR

## Key concepts demonstrated

- Digital modulation (16-QAM) and constellation mapping
- OFDM subcarrier multiplexing via IFFT/FFT
- Cyclic prefix insertion (inter-symbol interference mitigation)
- Wireless channel modeling: AWGN and Rayleigh flat fading
- Channel equalization (ideal CSI)
- Bit Error Rate (BER) performance analysis — the standard metric used to evaluate real communication systems

## How to run

**On Replit:** Just click **Run** — dependencies install automatically from `requirements.txt`.

**Locally:**
```bash
pip install -r requirements.txt
python main.py
```

## Output

Running the script prints per-SNR BER values to the console and saves two plots:

- `ber_vs_snr.png` — BER performance curve
- `constellation.png` — received signal constellation at 15 dB SNR

## System parameters

| Parameter | Value |
|---|---|
| Subcarriers (FFT size) | 64 |
| Cyclic Prefix length | 16 |
| Modulation | 16-QAM |
| OFDM symbols simulated | 500 per SNR point |
| SNR range | 0–24 dB |

These can be changed at the top of `main.py`.

## Possible extensions

- Add channel coding (convolutional / LDPC) for coded BER comparison
- Implement pilot-based channel estimation instead of ideal CSI
- Extend to a simple 2x2 MIMO configuration
- Support other modulation orders (QPSK, 64-QAM) for adaptive modulation demo

## Why this project

This project was built to demonstrate practical understanding of physical-layer wireless communication concepts (modulation, OFDM, fading channels, and performance analysis) that underpin real-world 4G/5G systems — relevant for roles in telecom, wireless R&D, and signal processing.
