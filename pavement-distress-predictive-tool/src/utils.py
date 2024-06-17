import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a specified CSV file path.
    
    Parameters:
    file_path (str): The path to the CSV file to be loaded.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def save_model(model, file_path: str):
    """
    Saves the trained model to a specified file path.
    
    Parameters:
    model: The model to be saved.
    file_path (str): The path where the model will be saved.
    """
    try:
        joblib.dump(model, file_path)
        print(f"Model saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving model to {file_path}: {e}")

def load_model(file_path: str):
    """
    Loads a model from a specified file path.
    
    Parameters:
    file_path (str): The path to the model file.
    
    Returns:
    The loaded model.
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: list = None) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame using a specified strategy.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing missing values.
    strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
    columns (list): List of columns to apply imputation on. If None, apply to all columns.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    """
    imputer = SimpleImputer(strategy=strategy)
    if columns:
        df[columns] = imputer.fit_transform(df[columns])
    else:
        df[:] = imputer.fit_transform(df)
    return df

def scale_features(df: pd.DataFrame, scaling_type: str = 'standard', columns: list = None) -> pd.DataFrame:
    """
    Scales features in the DataFrame using the specified scaling method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to scale.
    scaling_type (str): The type of scaling ('standard' or 'minmax').
    columns (list): List of columns to scale. If None, scale all columns.
    
    Returns:
    pd.DataFrame: The scaled DataFrame.
    """
    if scaling_type == 'standard':
        scaler = StandardScaler()
    elif scaling_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling type. Choose 'standard' or 'minmax'.")

    if columns:
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[:] = scaler.fit_transform(df)
    return df

def plot_correlation_matrix(df: pd.DataFrame, title: str = 'Correlation Matrix'):
    """
    Plots the correlation matrix for the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame whose correlation matrix is to be plotted.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.show()

def check_directory(path: str):
    """
    Checks if a directory exists and creates it if it does not.
    
    Parameters:
    path (str): The directory path to check.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists: {path}")

def save_dataframe(df: pd.DataFrame, file_path: str):
    """
    Saves the DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    file_path (str): The path to save the CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to {file_path}: {e}")

def summary_statistics(df: pd.DataFrame, columns: list = None):
    """
    Prints summary statistics for the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.
    columns (list): List of columns to include in the summary. If None, summarize all columns.
    """
    if columns:
        print(df[columns].describe())
    else:
        print(df.describe())

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Performs one-hot encoding on the specified columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to encode.
    columns (list): The list of columns to encode.
    
    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    try:
        df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
        print(f"One-hot encoding performed on columns: {columns}")
        return df_encoded
    except Exception as e:
        print(f"Error performing one-hot encoding: {e}")
        return df

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the DataFrame into train and test sets.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    target_column (str): The column to use as the target variable.
    test_size (float): The proportion of the data to include in the test set.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    X_train, X_test, y_train, y_test: The split data.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Data split into train and test sets with test size {test_size}")
    return X_train, X_test, y_train, y_test

def save_evaluation_metrics(metrics: dict, file_path: str):
    """
    Saves evaluation metrics to a specified file path.
    
    Parameters:
    metrics (dict): Dictionary containing evaluation metrics.
    file_path (str): The path to save the metrics file.
    """
    try:
        with open(file_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Evaluation metrics saved to {file_path}")
    except Exception as e:
        print(f"Error saving evaluation metrics to {file_path}: {e}")

if __name__ == "__main__":
    print("Utility functions for data processing and model management.")
