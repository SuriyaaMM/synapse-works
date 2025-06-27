import pandas as pd
import numpy as np
import os

def generate_large_csv(file_path: str, num_rows: int, num_features: int, num_labels: int):
    """
    Generates a large CSV file with random numerical features and labels.

    Args:
        file_path (str): The path where the CSV file will be saved.
        num_rows (int): The number of rows (samples) to generate.
        num_features (int): The number of feature columns.
        num_labels (int): The number of label columns.
    """
    print(f"Generating a CSV file with {num_rows:,} rows...")

    # Create feature column names
    feature_columns = [f'feature_{i}' for i in range(num_features)]
    # Create label column names
    label_columns = [f'label_{i}' for i in range(num_labels)]

    # Generate random features (float values between 0 and 100)
    features_data = np.random.rand(num_rows, num_features) * 100

    # Generate random labels
    # For labels, let's create a mix:
    # - Some can be binary (0 or 1) for multi-label classification
    # - Some can be integer class IDs for single-label classification
    # - Some can be float for regression targets
    labels_data = np.zeros((num_rows, num_labels))
    for i in range(num_labels):
        if i % 3 == 0: # Every 3rd label column, make it binary (0 or 1)
            labels_data[:, i] = np.random.randint(0, 2, size=num_rows)
        elif i % 3 == 1: # Next, make it an integer class ID (e.g., 0-9)
            labels_data[:, i] = np.random.randint(0, 10, size=num_rows)
        else: # The rest are float for regression
            labels_data[:, i] = np.random.rand(num_rows) * 10

    # Combine into a DataFrame
    df = pd.DataFrame(data=features_data, columns=feature_columns)
    df_labels = pd.DataFrame(data=labels_data, columns=label_columns)
    df = pd.concat([df, df_labels], axis=1)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"CSV file '{file_path}' with {num_rows:,} rows created successfully.")

# --- Configuration for your large dataset ---
output_csv_filename = 'large_test_dataset.csv'
desired_rows = 1_00_000  # 1 million rows (adjust as needed, e.g., 10_000 for small, 5_000_000 for very large)
num_features = 10      # Number of feature columns
num_labels = 5         # Number of label columns

# Generate the CSV
generate_large_csv(output_csv_filename, desired_rows, num_features, num_labels)