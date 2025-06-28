import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

def generate_regression_dataset(n_samples=1000, n_features=3, noise=20, filename="data/regression_dataset.csv"):
    """
    Generates a synthetic regression dataset and saves it to a CSV file.

    Args:
        n_samples (int): The number of samples (rows) to generate.
        n_features (int): The number of features (independent variables) to generate.
        noise (float): The standard deviation of the gaussian noise applied to the output.
        filename (str): The name of the CSV file to save the dataset to.
    """

    print(f"Generating a regression dataset with {n_samples} samples and {n_features} features...")

    # Generate synthetic regression data
    # X: features, y: target variable
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,  # All features are informative
        noise=noise,
        random_state=42  # for reproducibility
    )

    # Create column names for features
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

    # Create a Pandas DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y  # Add the target variable

    # Save the DataFrame to a CSV file
    try:
        df.to_csv(filename, index=False)
        print(f"Dataset successfully saved to '{filename}'")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Example usage:
    # Generate a dataset with 500 samples, 2 features, and a noise level of 15
    generate_regression_dataset(n_samples=500, n_features=2, noise=15, filename="../data/regression_test.csv")

    # Generate another dataset with default parameters
    generate_regression_dataset()