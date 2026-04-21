import pandas as pd
import numpy as np

import sqlite3

def clean_data(db_path):
    """
    Loads data directly from SQLite Database, drops duplicates, and handles missing values.
    Returns the cleaned DataFrame.
    """
    print("Connecting to SQLite database at:", db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM accidents", conn)
    conn.close()
    
    initial_shape = df.shape
    print(f"Initial raw data shape: {initial_shape}")
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Handle missing values
    # For categorical columns, we can fill missing with the mode (most common value)
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"Found missing values in columns: {missing_cols}")
        for col in missing_cols:
            mode_val = df[col].mode()[0]
            df.fillna({col: mode_val}, inplace=True)
            print(f"  - Filled missing values in '{col}' with median/mode = {mode_val}")
    
    print(f"Data shape after cleaning: {df.shape}")
    print("Data cleaning complete.\n")
    return df

def preprocess_features(df):
    """
    Prepares features for exploratory analysis and ML modeling.
    Since we are using tree-based models later, simple ordinal/nominal
    encoding is sufficient. We will separate features (X) and target (y).
    """
    print("Preprocessing data for machine learning...")
    
    # Target variable: accident_severity
    # Drop irrelevant columns like index
    if 'accident_index' in df.columns:
        df = df.drop(columns=['accident_index'])
        
    # All columns are currently numeric codes in STATS19 format,
    # so we don't need heavy dummy encoding for Random Forest, but 
    # ensuring they are correctly typed is good practice.
    
    print("Data preprocessing complete.\n")
    return df

if __name__ == "__main__":
    # Test block
    df_raw = clean_data("dataset/road_accidents.csv")
    df_processed = preprocess_features(df_raw)
    print(df_processed.head())
