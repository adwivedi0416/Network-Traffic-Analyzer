# 🛡️ Network Traffic Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A scalable machine learning pipeline that analyzes network traffic patterns and detects anomalies — including DDoS, port scans, probe attacks, and unauthorized access attempts — using an ensemble of **Random Forest** and **XGBoost** classifiers.

---

## 🎯 Key Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| Detection Accuracy | **92.4%** | **93.1%** |
| F1 Score (weighted) | **91.8%** | **92.6%** |
| ROC-AUC | **0.974** | **0.981** |
| False Positive Rate | **~4.2%** | **~3.8%** |

> Benchmarked on NSL-KDD dataset. Ensemble inference combines both models for higher confidence scoring.

---

## 🏗️ Architecture

```
Raw Traffic / PCAP
        ↓
  Feature Extraction
  (41 packet + flow features)
        ↓
  Preprocessing Pipeline
  (Encoding → Scaling)
        ↓
  ┌─────────────┐    ┌──────────┐
  │ Random Forest│    │ XGBoost  │
  └─────────────┘    └──────────┘
        └────────┬────────┘
          Ensemble Vote
               ↓
    Prediction + Risk Score
    (NORMAL / ATTACK + confidence %)
```

---

## 📁 Project Structure

```
network-traffic-analyzer/
├── src/
│   └── pipeline.py         # Core ML pipeline
├── models/                 # Saved models (after training)
├── data/                   # Place your NSL-KDD dataset here
├── results/                # Evaluation outputs
├── train.py                # Training script
├── app.py                  # Flask REST API + dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/network-traffic-analyzer.git
cd network-traffic-analyzer
pip install -r requirements.txt
```

### 2. Train (demo — uses synthetic data)
```bash
python train.py
```

### 3. Train with real NSL-KDD data
```bash
# Download: https://www.unb.ca/cic/datasets/nsl.html
python train.py --data path/to/KDDTrain+.txt
```

### 4. Start the API server
```bash
python app.py
# Open http://localhost:5000
```

---

## 🔌 API Usage

**Classify a single traffic record:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":232,"dst_bytes":8153,"logged_in":1,"count":8,"serror_rate":0.0,"same_srv_rate":1.0}'
```

**Response:**
```json
{
  "prediction": "NORMAL",
  "confidence": 97.4,
  "risk_level": "LOW",
  "rf_confidence": 96.8,
  "xgb_confidence": 98.1
}
```

---

## 🔬 Attack Categories Detected

| Category | Examples |
|---|---|
| **DoS** | neptune, smurf, back, teardrop |
| **Probe** | ipsweep, portsweep, nmap, satan |
| **R2L** | guess_passwd, ftp_write, warezmaster |
| **U2R** | buffer_overflow, rootkit, loadmodule |

---

## 📊 Dataset

Designed for **NSL-KDD** — improved version of KDD Cup 99.
- Download: [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
- 41 packet/flow-level features per connection record
- Built-in synthetic generator included for quick demos

---

## 🛠️ Tech Stack
**ML:** scikit-learn · XGBoost &nbsp;|&nbsp; **Data:** NumPy · Pandas &nbsp;|&nbsp; **API:** Flask · joblib

---

## 📄 License
MIT License — see [LICENSE](LICENSE)
