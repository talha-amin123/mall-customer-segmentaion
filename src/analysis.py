"""
Module for analysis and interpretation of clustering results
"""
import pandas as pd


def get_cluster_statistics(original_data: pd.DataFrame, cluster_labels: list) -> pd.DataFrame:
    """
    Calculate mean statistics for numerical features by cluster
    
    Args:
        original_data (pd.DataFrame): Original customer data
        cluster_labels (list): Cluster assignments
        
    Returns:
        pd.DataFrame: Mean values of features by cluster
    """
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Calculate mean of numerical features
    numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    cluster_means = data_with_clusters.loc[:, numerical_features + ['Cluster']].groupby('Cluster').mean()
    
    return cluster_means


def get_cluster_gender_distribution(original_data: pd.DataFrame, cluster_labels: list) -> pd.DataFrame:
    """
    Calculate gender distribution by cluster
    
    Args:
        original_data (pd.DataFrame): Original customer data
        cluster_labels (list): Cluster assignments
        
    Returns:
        pd.DataFrame: Gender distribution (proportions) by cluster
    """
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    gender_dist = data_with_clusters.loc[:, ['Gender', 'Cluster']].groupby('Cluster').value_counts(normalize=True).unstack(fill_value=0)
    
    return gender_dist


def get_cluster_summary(original_data: pd.DataFrame, cluster_labels: list) -> pd.DataFrame:
    """
    Get comprehensive cluster summary combining numerical and categorical features
    
    Args:
        original_data (pd.DataFrame): Original customer data
        cluster_labels (list): Cluster assignments
        
    Returns:
        pd.DataFrame: Summary statistics for each cluster
    """
    cluster_means = get_cluster_statistics(original_data, cluster_labels)
    gender_dist = get_cluster_gender_distribution(original_data, cluster_labels)
    
    summary = pd.merge(cluster_means, gender_dist, left_index=True, right_index=True)
    
    # Rename gender columns for clarity
    if 'Female' in summary.columns:
        summary.rename(columns={'Female': 'Female Ratio'}, inplace=True)
    if 'Male' in summary.columns:
        summary.rename(columns={'Male': 'Male Ratio'}, inplace=True)
    
    return summary


def get_cluster_ranges(original_data: pd.DataFrame, cluster_labels: list) -> pd.DataFrame:
    """
    Calculate min and max ranges for numerical features by cluster
    
    Args:
        original_data (pd.DataFrame): Original customer data
        cluster_labels (list): Cluster assignments
        
    Returns:
        pd.DataFrame: Min and max values of features by cluster
    """
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Calculate min and max for each feature by cluster
    range_stats = pd.DataFrame()
    for feature in numerical_features:
        min_vals = data_with_clusters.groupby('Cluster')[feature].min()
        max_vals = data_with_clusters.groupby('Cluster')[feature].max()
        
        range_stats[f'{feature} (Min)'] = min_vals
        range_stats[f'{feature} (Max)'] = max_vals
    
    return range_stats


def interpret_clusters(summary: pd.DataFrame, ranges: pd.DataFrame = None) -> dict:
    """
    Generate interpretations of each cluster based on statistics
    
    Args:
        summary (pd.DataFrame): Cluster summary statistics
        ranges (pd.DataFrame): Min/max ranges for features by cluster (optional)
        
    Returns:
        dict: Interpretations for each cluster
    """
    interpretations = {}
    
    for cluster_id in summary.index:
        cluster_data = summary.loc[cluster_id]
        
        age = cluster_data.get('Age', 0)
        income = cluster_data.get('Annual Income (k$)', 0)
        spending = cluster_data.get('Spending Score (1-100)', 0)
        
        # Create interpretation based on characteristics
        if spending > 60 and income > 60:
            profile = "High-Value Customer (High Income, High Spending)"
        elif spending < 40 and income < 40:
            profile = "Budget-Conscious Customer (Low Income, Low Spending)"
        elif spending > 60 and income < 50:
            profile = "Affordable Luxury Seeker (Low Income, High Spending)"
        elif spending < 40 and income > 60:
            profile = "Cautious Affluent (High Income, Low Spending)"
        else:
            profile = "Moderate Customer (Average Income and Spending)"
        
        interpretation = {
            'profile': profile,
            'avg_age': round(age, 2),
            'avg_income': round(income, 2),
            'avg_spending_score': round(spending, 2)
        }
        
        # Add ranges if provided
        if ranges is not None and cluster_id in ranges.index:
            range_data = ranges.loc[cluster_id]
            interpretation['age_range'] = (int(range_data['Age (Min)']), int(range_data['Age (Max)']))
            interpretation['income_range'] = (round(range_data['Annual Income (k$) (Min)'], 1), 
                                             round(range_data['Annual Income (k$) (Max)'], 1))
            interpretation['spending_range'] = (int(range_data['Spending Score (1-100) (Min)']), 
                                               int(range_data['Spending Score (1-100) (Max)']))
        
        interpretations[cluster_id] = interpretation
    
    return interpretations
