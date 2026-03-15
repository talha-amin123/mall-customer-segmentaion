"""
Module for loading and basic data exploration
"""
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load customer data from CSV file
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Customer data
    """
    data = pd.read_csv(filepath)
    return data


def get_data_info(data: pd.DataFrame) -> None:
    """
    Display basic information about the dataset
    
    Args:
        data (pd.DataFrame): Customer data
    """
    print("Dataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())
    print("\nFirst few rows:")
    print(data.head())
