import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts new features from the existing dataset.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with new features.
    """
    try:
        # Convert severity levels to numeric for easier processing
        severity_map = {'L': 1, 'M': 2, 'H': 3}
        df['Severity Numeric'] = df['Severity'].map(severity_map)
        
        # Calculate distress density (Quantity per area)
        df['Distress Density'] = df['Quantity'] / df['True Area (Ft2)']
        
        # Flag for critical PCI (e.g., PCI < 40 might be critical)
        df['Critical PCI'] = df['PCI'] < 40

        print("Feature extraction successful.")
        return df
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return df

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the existing features to improve the dataset.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The transformed DataFrame.
    """
    try:
        # Log-transform Quantity to reduce skewness (assuming Quantity is always positive)
        df['Log Quantity'] = np.log1p(df['Quantity'])

        print("Feature transformation successful.")
        return df
    except Exception as e:
        print(f"Error during feature transformation: {e}")
        return df

def scale_features(df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    """
    Scales numerical features using standard scaling (mean=0, std=1).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_scale (list): List of column names to be scaled.
    
    Returns:
    pd.DataFrame: The DataFrame with scaled features.
    """
    try:
        # Ensure the columns to be scaled are of float type to avoid dtype issues
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = df_scaled[columns_to_scale].astype(float)
        
        scaler = StandardScaler()
        df_scaled.loc[:, columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        
        print("Feature scaling successful.")
        return df_scaled
    except Exception as e:
        print(f"Error during feature scaling: {e}")
        return df

def save_features(df: pd.DataFrame, file_path: str):
    """
    Saves the DataFrame with engineered features to a specified CSV file path.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with engineered features.
    file_path (str): The path to save the DataFrame.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Engineered features saved to {file_path}")
    except Exception as e:
        print(f"Error saving features to {file_path}: {e}")
