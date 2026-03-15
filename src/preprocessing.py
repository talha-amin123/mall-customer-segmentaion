"""
Module for data preprocessing and feature scaling
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess data by removing CustomerID and Gender columns, then scaling
    
    Args:
        data (pd.DataFrame): Customer data
        
    Returns:
        np.ndarray: Scaled feature data
    """
    scaler = StandardScaler()
    # Remove CustomerID and Gender columns (keep numerical features)
    numerical_features = data.iloc[:, 2:]
    scaled_data = scaler.fit_transform(numerical_features)
    return scaled_data
