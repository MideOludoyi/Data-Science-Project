import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

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

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (str): The target column for prediction.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    tuple: Tuple containing training and testing sets for features and target.
    """
    try:
        # Debugging output to ensure correct input types and content
        print("Checking DataFrame type and columns...")
        print("DataFrame type:", type(df))
        print("DataFrame columns:", df.columns)
        
        print("Checking target type and value...")
        print("Target type:", type(target))
        print("Target value:", target)
        
        # Check if target is a string and if it exists in the DataFrame columns
        if not isinstance(target, str):
            raise ValueError(f"Target should be a string representing the column name, but got {type(target)}.")
        
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in DataFrame columns: {df.columns.tolist()}")
        
        # Proceed with splitting the data
        X = df.drop(columns=[target])
        y = df[target]
        
        print("Features (X) preview:")
        print(X.head())
        
        print("Target (y) preview:")
        print(y.head())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print("Data successfully split into training and testing sets.")
        return X_train, X_test, y_train, y_test
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except KeyError as ke:
        print(f"KeyError: {ke}")
    except Exception as e:
        print(f"Error splitting data: {e}")
    return None, None, None, None
    
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor model.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.
    
    Returns:
    RandomForestRegressor: The trained model.
    """
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model training completed.")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model on the test set.
    
    Parameters:
    model (RandomForestRegressor): The trained model.
    X_test (pd.DataFrame): The testing features.
    y_test (pd.Series): The testing target.
    
    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model evaluation metrics: RMSE={rmse}, R2={r2}")
        return {"RMSE": rmse, "R2": r2}
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}

def save_model(model: RandomForestRegressor, file_path: str):
    """
    Saves the trained model to a specified file path.
    
    Parameters:
    model (RandomForestRegressor): The trained model.
    file_path (str): The path to save the model file.
    """
    try:
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")