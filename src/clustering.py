"""
Module for K-Means clustering and segmentation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def find_optimal_clusters(data: pd.DataFrame, k_range: range = range(1, 11)) -> tuple:
    """
    Find optimal number of clusters using elbow method
    
    Args:
        data (pd.DataFrame): PCA-transformed data
        k_range (range): Range of cluster numbers to test
        
    Returns:
        tuple: (wcss list, kmeans models list)
    """
    wcss = []
    kmeans_models = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        kmeans_models.append(kmeans)
    
    return wcss, kmeans_models


def plot_elbow_curve(wcss: list, k_range: range = range(1, 11), save_path: str = None) -> None:
    """
    Plot elbow curve for optimal k selection
    
    Args:
        wcss (list): Within-cluster sum of squares
        k_range (range): Range of cluster numbers
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def perform_clustering(data: pd.DataFrame, n_clusters: int = 3) -> tuple:
    """
    Perform K-Means clustering on data
    
    Args:
        data (pd.DataFrame): PCA-transformed data
        n_clusters (int): Number of clusters
        
    Returns:
        tuple: (kmeans object, cluster labels, data with cluster labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(data)
    
    clustered_data = data.copy()
    clustered_data['Cluster'] = clusters
    
    return kmeans, clusters, clustered_data


def plot_clusters(data: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot customer segments in 2D PCA space
    
    Args:
        data (pd.DataFrame): PCA data with cluster labels
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='Cluster',
        data=data,
        palette='Set1',
        s=100
    )
    plt.title('Customer Segmentation using K-Means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()
