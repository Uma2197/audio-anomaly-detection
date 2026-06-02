# Audio Anomaly Detection Pipeline
### Industrial Predictive Maintenance — End-to-End Demo

---

## Overview
This notebook demonstrates a complete anomaly detection pipeline 
applied to industrial machine audio data — directly relevant to 
use cases like OHT vehicle health monitoring in semiconductor 
manufacturing.

## Pipeline
Raw Audio → Waveform → Spectrogram → MFCCs → Feature Extraction
→ Multimodal Fusion → Autoencoder → Temporal Drift Validation

## Key Results
- Autoencoder: **0/40 false alarms, 10/10 anomalies detected**
- Multimodal fusion: **2.4× better separation** than audio alone
- Temporal drift analysis: degradation detectable **weeks before failure**

## Sections
- **Section 1** — Audio pipeline: waveform, spectrogram, MFCCs, feature extraction
- **Section 2** — Anomaly detection: Isolation Forest vs Autoencoder
- **Section 3** — Multimodal fusion: audio + velocity + acceleration + maintenance logs
- **Section 4** — Validation without labels: 4 strategies including temporal drift

## Requirements
pip install librosa scikit-learn matplotlib numpy

## Context
Simulated dataset based on real industrial use case — 50 recordings 
(40 normal, 10 abnormal), 29 features per recording combining 
MFCC statistics with trajectory and maintenance data.
