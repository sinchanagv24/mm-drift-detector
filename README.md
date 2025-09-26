# 📊 Multimodal Data Drift Detector

⚡ A Streamlit app to detect **data drift**, ingest **context** (release notes + screenshots), and explain **why** metrics changed.

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
  - Threshold tuning, cost-sensitive metrics, reproducible configs & reports  

---

## 🎥 Demo Video
[![Demo Video](https://img.youtube.com/vi/BjrNhvkNXq4/0.jpg)](https://youtu.be/BjrNhvkNXq4)

---

## 📂 Installation

```bash
git clone https://github.com/sinchanagv24/mm-drift-detector.git
cd mm-drift-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
