"""
Module for PCA analysis and dimensionality reduction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def perform_pca_analysis(scaled_data: np.ndarray) -> tuple:
    """
    Perform PCA analysis to determine optimal number of components
    
    Args:
        scaled_data (np.ndarray): Scaled feature data
        
    Returns:
        tuple: (pca object, explained_variance_ratio)
    """
    pca = PCA()
    pca.fit(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    return pca, explained_variance


def plot_explained_variance(explained_variance: np.ndarray, save_path: str = None) -> None:
    """
    Plot cumulative explained variance
    
    Args:
        explained_variance (np.ndarray): Explained variance ratios
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def apply_pca(scaled_data: np.ndarray, n_components: int = 2) -> tuple:
    """
    Apply PCA with specified number of components
    
    Args:
        scaled_data (np.ndarray): Scaled feature data
        n_components (int): Number of principal components
        
    Returns:
        tuple: (pca object, transformed data as DataFrame)
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(
        data=pca_data,
        columns=[f'Principal Component {i+1}' for i in range(n_components)]
    )
    return pca, pca_df


def get_pca_loadings(pca: PCA, feature_names: list = None) -> pd.DataFrame:
    """
    Get PCA loadings (components)
    
    Args:
        pca (PCA): Fitted PCA object
        feature_names (list, optional): Names of original features
        
    Returns:
        pd.DataFrame: PCA loadings
    """
    if feature_names is None:
        feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    return loadings
