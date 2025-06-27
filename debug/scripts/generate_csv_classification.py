import pandas as pd
import numpy as np
import os

def generate_large_csv_for_classification(file_path: str, num_rows: int, num_features: int, num_classes: int):
    """
    Generates a large CSV file with random numerical features and a single label column
    for multi-class classification.

    Args:
        file_path (str): The path where the CSV file will be saved.
        num_rows (int): The number of rows (samples) to generate.
        num_features (int): The number of feature columns.
        num_classes (int): The number of distinct classes for the label (e.g., 10 for CIFAR-10).
                           Labels will be integers from 0 to num_classes-1.
    """
    print(f"Generating a classification CSV file with {num_rows:,} rows...")

    # Create feature column names
    feature_columns = [f'feature_{i}' for i in range(num_features)]

    # Generate random features (float values between 0 and 100)
    features_data = np.random.rand(num_rows, num_features) * 100

    # Generate labels for classification (single column of integer class IDs)
    # Labels will be integers from 0 to num_classes - 1
    # We explicitly create a 2D array for the labels, even if it's just one column,
    # to maintain consistency for DataFrame creation.
    labels_data = np.random.randint(0, num_classes, size=(num_rows, 1))

    # Define the single label column name
    label_column_name = 'class_label'
    
    # Combine into a DataFrame
    df = pd.DataFrame(data=features_data, columns=feature_columns)
    df_labels = pd.DataFrame(data=labels_data, columns=[label_column_name])
    df = pd.concat([df, df_labels], axis=1)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Classification CSV file '{file_path}' with {num_rows:,} rows created successfully.")

# --- Configuration for your large classification dataset ---
output_csv_filename_cls = 'large_test_classification_dataset.csv'
desired_rows_cls = 100_000  # 100,000 rows
num_features_cls = 10      # Number of feature columns
num_classes_cls = 10       # Number of distinct classes (e.g., for CIFAR-10, MNIST-like)

# Generate the CSV for classification
generate_large_csv_for_classification(output_csv_filename_cls, desired_rows_cls, num_features_cls, num_classes_cls)
