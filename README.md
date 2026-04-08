# LFB Response Time Prediction: AI in Public Policy

This repository contains the replication code for a predictive study on the London Fire Brigade (LFB) response times. The project demonstrates the application of Machine Learning in public resource allocation and emergency dispatching.

## 1. Data Repository & Execution Strategy
* **Official Data Source:** Greater London Authority (GLA) - London Datastore
* **Dataset Name:** London Fire Brigade Mobilisation Records

**Bulletproof Single-Execution Design:** To strictly adhere to the course's "single execution" requirement and ensure 100% compatibility across any environment (e.g., a clean PyCharm installation or command prompt), I have implemented two fail-safes:
1. **Embedded Data:** A representative micro-sample of 100 real records is embedded directly within the `main.py` script using `io.StringIO`. This completely eliminates local path configuration errors. *(Note: A standalone `lfb_sample.csv` is kept in this repository purely for structural reference).*
2. **Auto-Dependency Installer:** The script features a pre-flight checker. If the evaluator's environment is missing required libraries (`pandas`, `scikit-learn`), the script will automatically install them via standard `pip` before executing the core model.

## 2. Model Overview
The script implements a baseline **Multiple Linear Regression** model. It predicts the emergency attendance time (in seconds) based on the hour of the call and spatial coordinates (Easting/Northing).

## 3. How to Run
1. Download or simply copy the `main.py` script to any local IDE (like PyCharm) or execute it via terminal.
2. Run the script:
   ```bash
   python main.py
