# ==========================================
# Replication Code: EMS Response Time Prediction
# Dataset: London Fire Brigade Mobilisation Records
# Model: Multiple Linear Regression
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main():
    print("Step 1: Loading Sample Data...")
    # Load the local sample dataset to ensure single-execution
    # The full 400MB dataset is available at London Datastore.
    try:
        df = pd.read_csv('lfb_sample.csv')
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Error: lfb_sample.csv not found. Please ensure it is in the same directory.")
        return

    print("Step 2: Data Preprocessing...")
    # Drop rows with missing attendance times (Data Cleaning)
    df = df.dropna(subset=['AttendanceTimeSeconds'])
    
    # Define features (X) and target variable (y)
    X = df[['HourOfCall', 'Easting_m', 'Northing_m']]
    y = df['AttendanceTimeSeconds']

    print("Step 3: Splitting data into Training and Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 4: Training the Linear Regression Model...")
    # Initialize and train the baseline model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Step 5: Evaluating the Model...")
    # Predict on the test set
    predictions = model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Evaluation Complete.")
    print(f"Mean Absolute Error (MAE): {mae:.2f} seconds")
    print("Conclusion: The baseline model can estimate general trends, but further complex modeling is needed for high-stakes dispatching.")

if __name__ == "__main__":
    main()
