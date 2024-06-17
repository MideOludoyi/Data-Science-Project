import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(file_path: str):
    """
    Loads the trained model from the specified file path.
    
    Parameters:
    file_path (str): The path to the model file.
    
    Returns:
    object: The loaded model.
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a specified CSV file path.
    
    Parameters:
    file_path (str): The path to the CSV file to be loaded.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates the model performance using various metrics.
    
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    
    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"Evaluation Metrics:\nRMSE: {rmse}\nMAE: {mae}\nR2 Score: {r2}")
        return {"RMSE": rmse, "MAE": mae, "R2": r2}
    except Exception as e:
        print(f"Error evaluating model performance: {e}")
        return {}

def plot_results(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plots the results to visualize the performance of the model.
    
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    """
    try:
        plt.figure(figsize=(14, 7))

        # Scatter plot for actual vs predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('Actual PCI')
        plt.ylabel('Predicted PCI')
        plt.title('Actual vs Predicted PCI')

        # Residual plot
        residuals = y_true - y_pred
        plt.subplot(1, 2, 2)
        sns.histplot(residuals, kde=True, color='blue')
        plt.xlabel('Residuals')
        plt.title('Residual Distribution')

        plt.tight_layout()
        plt.show()
        
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")

def save_evaluation_metrics(metrics: dict, file_path: str):
    """
    Saves the evaluation metrics to a specified file path.
    
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

