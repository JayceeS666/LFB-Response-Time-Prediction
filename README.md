# LFB Response Time Prediction: AI in Public Policy

This repository contains the replication code for a predictive study on the London Fire Brigade (LFB) response times. The project demonstrates the application of Machine Learning in public resource allocation and emergency dispatching.

## 1. Data Repository (Data Source)
* **Official Data Source:** Greater London Authority (GLA) - London Datastore
* **Dataset Name:** London Fire Brigade Mobilisation Records
* **Full Data Download Link:** [London Datastore LFB Data](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)

**Important Note on Data Hosting & Execution:** Due to the massive size of the complete historical dispatch data (approx. 400MB), the raw CSV file is not hosted on GitHub. To strictly adhere to the **"single execution"** requirement and ensure 100% compatibility across any IDEs (like PyCharm) without local path configuration errors, a representative micro-sample of 100 real records is **embedded directly within the `main.py` script** using `io.StringIO`. 

*(Note: A standalone `lfb_sample.csv` is also kept in this repository purely for structural reference, but the script does not depend on it for execution.)*

## 2. Model and Code
The repository contains a single Python script (`main.py`) that implements a baseline **Multiple Linear Regression** model. It predicts the emergency attendance time (in seconds) based on the hour of the call and spatial coordinates (Easting/Northing).

## 3. How to Run (Bulletproof Single Execution)
The code is designed to be fully self-contained.
1. Ensure you have Python installed, along with the `pandas` and `scikit-learn` libraries.
2. Copy the `main.py` code into any IDE (e.g., PyCharm, VS Code) or download the file.
3. Simply run the script. It will automatically load the embedded sample data from its own memory, preprocess the variables, train the baseline model, and output the predictive Mean Absolute Error (MAE: 70.20 seconds).
