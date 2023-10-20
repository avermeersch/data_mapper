import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def normalize_columns(df):
    """
    Normalize column names of a DataFrame: convert to lowercase and replace spaces with underscores.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns to normalize.
        
    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df