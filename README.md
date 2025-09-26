# 📊 Multimodal Data Drift Detector

⚡ A Streamlit app to detect **data drift**, ingest **context** (release notes + screenshots), and explain **why** metrics changed.

---

## 📜 Project Description
Dashboards and models often fail silently when data drifts: null spikes, schema changes, or category shifts make KPIs unreliable. Traditional drift tools only flag what changed, leaving teams guessing why.  

The **Multimodal Data Drift Detector** combines statistical drift detection with context ingestion from release notes and dashboard screenshots. It detects null %, numeric drift (KS/mean shift), categorical drift (PSI), and outliers, then ingests PDFs and screenshots to capture signals like “currency switched USD→EUR” or “API v2 deployed.”  

The system generates a plain-English narrative with actionable next steps such as validating ETL, backfilling data, or updating contracts. A simulated chat integration enables instant alerts and team responses. An optional sandbox supports threshold tuning, cost-sensitive metrics, and reproducibility with downloadable configs.  

This helps teams move beyond detection to diagnosis and action, reducing downtime and keeping metrics trustworthy.  

---

## 🚀 Features
- **Drift & Anomaly Checks**  
  - Null % by column  
  - KS test & mean shift for numeric drift  
  - PSI for categorical drift  
  - Outlier detection (z-scores)  
- **Context Ingestion**  
  - Extracts signals from PDF/text release notes  
  - OCR on dashboard screenshots  
- **Narrative Output**  
  - Human-readable explanation with root-cause hypotheses and suggested actions  
- **Collaboration**  
  - Send alerts to a simulated chat log (Slack/Discord integration planned)  
- **Optional Sandbox**  
  - Threshold tuning  
  - Cost-sensitive metrics  
  - Reproducibility with downloadable configs & reports  

---

## 🎥 Demo Video
[![Demo Video](https://img.youtube.com/vi/BjrNhvkNXq4/0.jpg)](https://youtu.be/BjrNhvkNXq4)

---

## 🤝 Partner Tools
- **Snowflake** – data warehouse integration (planned connector)  
- **Streamlit** – rapid prototyping UI  
- **scikit-learn / imbalanced-learn** – drift detection and ML utilities  
- **pdfplumber / pytesseract** – context extraction from PDFs and images  

---

## 📂 Installation
Clone the repository and set up the environment:

```bash
git clone https://github.com/sinchanagv24/mm-drift-detector.git
cd mm-drift-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---
## Running Streamlit app

```bash
streamlit run app.py
```
---
