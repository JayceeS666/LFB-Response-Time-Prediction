# LFB Response Time Prediction: AI in Public Policy

This repository contains the replication code for a predictive study on the London Fire Brigade (LFB) response times. The project demonstrates the application of Machine Learning in public resource allocation and emergency dispatching.

## 1. Data Repository (Data Source)
* **Official Data Source:** Greater London Authority (GLA) - London Datastore
* **Dataset Name:** London Fire Brigade Mobilisation Records
* **Full Data Download Link:** [London Datastore LFB Data](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)

**Important Note for Execution:** Due to the massive size of the complete historical dispatch data (approx. 400MB), the raw CSV file is not hosted on GitHub to ensure seamless execution. Instead, a representative micro-sample (`lfb_sample.csv`) containing real data structures is provided in this repository. This guarantees that the replication code can be executed instantly without heavy local downloads.

## 2. Model and Code
The repository contains a single Python script (`main.py`) that implements a baseline **Multiple Linear Regression** model. It predicts the emergency attendance time (in seconds) based on the hour of the call and spatial coordinates (Easting/Northing).

## 3. How to Run (Single Execution Requirement)
The code is designed to strictly meet the "single execution" requirement. 
1. Ensure you have Python installed, along with the `pandas` and `scikit-learn` libraries.
2. Clone or download this repository to your local machine (ensure `main.py` and `lfb_sample.csv` are in the same folder).
3. Open your terminal or command prompt, navigate to the folder, and run:
   ```bash
   python main.py
