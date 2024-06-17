import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a specified CSV file path into a DataFrame.
    
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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the pavement distress dataset.
    
    This function:
    1. Converts necessary columns to strings if they are not already.
    2. Removes commas from numerical columns represented as strings.
    3. Converts cleaned string representations of numbers back to floats.
    4. Fills missing values in numerical and categorical columns.
    
    Parameters:
    df (pd.DataFrame): The raw dataset.
    
    Returns:
    pd.DataFrame: The cleaned dataset.
    """
    try:
        # Ensure 'True Area (Ft2)' and 'Quantity' are strings
        df['True Area (Ft2)'] = df['True Area (Ft2)'].astype(str)
        df['Quantity'] = df['Quantity'].astype(str)

        # Remove commas and convert to float
        df['True Area (Ft2)'] = df['True Area (Ft2)'].str.replace(',', '').astype(float)
        df['Quantity'] = df['Quantity'].str.replace(',', '').astype(float)

        # Impute missing values for 'Quantity' using median
        imputer_quantity = SimpleImputer(strategy='median')
        df['Quantity'] = imputer_quantity.fit_transform(df[['Quantity']])

        # Impute missing values for categorical columns using most frequent
        imputer_mode = SimpleImputer(strategy='most_frequent')
        df['Distress Description'] = imputer_mode.fit_transform(df[['Distress Description']].values.reshape(-1, 1)).ravel()
        df['Quantity Units'] = imputer_mode.fit_transform(df[['Quantity Units']].values.reshape(-1, 1)).ravel()
        df['Severity'] = imputer_mode.fit_transform(df[['Severity']].values.reshape(-1, 1)).ravel()

        print("Data cleaning successful.")
        return df

    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return df

def save_cleaned_data(df: pd.DataFrame, file_path: str):
    """
    Saves the cleaned DataFrame to a specified CSV file path.
    
    Parameters:
    df (pd.DataFrame): The cleaned dataset.
    file_path (str): The path to save the cleaned data.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")