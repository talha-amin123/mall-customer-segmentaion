"""
Main script for Mall Customer Segmentation Analysis
Orchestrates the entire pipeline from data loading to final results
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.data_loader import load_data, get_data_info
from src.preprocessing import preprocess_data
from src.pca_analysis import perform_pca_analysis, apply_pca, get_pca_loadings
from src.clustering import find_optimal_clusters, perform_clustering
from src.analysis import get_cluster_summary, get_cluster_ranges, interpret_clusters


def plot_combined_analysis(explained_variance, wcss, clustered_pca_df):
    """
    Create a combined figure with all analysis plots in a 2x2 grid
    
    Args:
        explained_variance (np.ndarray): PCA explained variance ratios
        wcss (list): Within-cluster sum of squares for different k values
        clustered_pca_df (pd.DataFrame): PCA data with cluster assignments
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Mall Customer Segmentation Analysis - Complete Results', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Cumulative Explained Variance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.cumsum(explained_variance), marker='o', linestyle='--', 
             linewidth=2.5, markersize=8, color='#2E86AB')
    ax1.fill_between(range(len(explained_variance)), np.cumsum(explained_variance), 
                      alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax1.set_title('PCA: Explained Variance vs. Number of Components', 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Elbow Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, 11), wcss, marker='o', linestyle='-', linewidth=2.5, 
             markersize=8, color='#A23B72')
    ax2.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Optimal k=3', alpha=0.7)
    ax2.scatter([3], [wcss[2]], color='red', s=200, zorder=5, marker='*')
    ax2.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax2.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    ax2.set_title('Elbow Method For Optimal k Selection', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Customer Segments (2D Scatter)
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for cluster in sorted(clustered_pca_df['Cluster'].unique()):
        cluster_data = clustered_pca_df[clustered_pca_df['Cluster'] == cluster]
        ax3.scatter(cluster_data['Principal Component 1'],
                   cluster_data['Principal Component 2'],
                   label=f'Cluster {cluster}',
                   s=150, alpha=0.7, color=colors[cluster], edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Principal Component 1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Principal Component 2', fontsize=12, fontweight='bold')
    ax3.set_title('Customer Segmentation using K-Means Clustering', 
                  fontsize=13, fontweight='bold', pad=10)
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Individual Component Variance
    ax4 = fig.add_subplot(gs[1, 1])
    components = range(1, min(len(explained_variance) + 1, 11))
    bars = ax4.bar(components, explained_variance[:10], alpha=0.8, 
                   color='#F18F01', edgecolor='black', linewidth=1.5)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_xlabel('Principal Components', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Individual Component Variance Contribution', 
                  fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_xticks(components)
    
    plt.show()


def main():
    """
    Main pipeline for customer segmentation analysis
    """
    print("="*80)
    print("MALL CUSTOMER SEGMENTATION ANALYSIS")
    print("="*80)
    
    # ===== STEP 1: Load Data =====
    print("\n[STEP 1] Loading Customer Data...")
    data = load_data('data/mall_customers.csv')
    print(f"✓ Loaded {len(data)} customer records")
    print(f"✓ Features: {list(data.columns)}")
    
    # ===== STEP 2: Data Exploration =====
    print("\n[STEP 2] Data Exploration...")
    get_data_info(data)
    
    # ===== STEP 3: Preprocessing =====
    print("\n[STEP 3] Preprocessing Data...")
    scaled_data = preprocess_data(data)
    print(f"✓ Data scaled to shape: {scaled_data.shape}")
    
    # ===== STEP 4: PCA Analysis =====
    print("\n[STEP 4] Performing PCA Analysis...")
    pca, explained_variance = perform_pca_analysis(scaled_data)
    cumulative_variance = explained_variance.cumsum()
    print(f"✓ Variance explained by 2 components: {cumulative_variance[1]:.2%}")
    print(f"✓ Variance explained by 3 components: {cumulative_variance[2]:.2%}")
    
    # ===== STEP 5: Apply PCA with 2 Components =====
    print("\n[STEP 5] Applying PCA with 2 Components...")
    pca_2d, pca_df = apply_pca(scaled_data, n_components=2)
    print(f"✓ Transformed data shape: {pca_df.shape}")
    
    # Get loadings
    loadings = get_pca_loadings(pca_2d)
    print("\nPCA Loadings (Component Contributions):")
    print(loadings)
    
    # ===== STEP 6: Find Optimal Clusters =====
    print("\n[STEP 6] Finding Optimal Number of Clusters (Elbow Method)...")
    wcss, kmeans_models = find_optimal_clusters(pca_df, k_range=range(1, 11))
    print(f"✓ Evaluated 1-10 clusters")
    
    # ===== STEP 7: Perform Clustering =====
    best_k = 3
    print(f"\n[STEP 7] Performing K-Means Clustering (k={best_k})...")
    kmeans, clusters, clustered_pca_df = perform_clustering(pca_df, n_clusters=best_k)
    print(f"✓ Clustering complete - Cluster distribution: {[list(clusters).count(i) for i in range(best_k)]}")
    
    # ===== STEP 8: Display Combined Analysis Plots =====
    print("\n[STEP 8] Generating Combined Analysis Plots...")
    plot_combined_analysis(explained_variance, wcss, clustered_pca_df)
    print("✓ Plots displayed successfully - Execution continuing...\n")
    # ===== STEP 9: Analysis and Interpretation =====
    print("[STEP 9] Analyzing Clusters...")
    cluster_summary = get_cluster_summary(data, clusters)
    cluster_ranges = get_cluster_ranges(data, clusters)
    
    print("\nCluster Summary Statistics (Mean Values):")
    print(cluster_summary)
    print("\n")
    
    print("Cluster Ranges (Min - Max Values):")
    print(cluster_ranges)
    print("\n")
    
    # Get interpretations
    interpretations = interpret_clusters(cluster_summary, cluster_ranges)
    print("CLUSTER PROFILES:")
    print("-" * 80)
    for cluster_id, info in interpretations.items():
        print(f"\nCluster {cluster_id}: {info['profile']}")
        print(f"  - Average Age: {info['avg_age']} years")
        print(f"    Range: {info['age_range'][0]} - {info['age_range'][1]} years")
        print(f"  - Average Annual Income: ${info['avg_income']:.1f}k")
        print(f"    Range: ${info['income_range'][0]:.1f}k - ${info['income_range'][1]:.1f}k")
        print(f"  - Average Spending Score: {info['avg_spending_score']:.1f}/100")
        print(f"    Range: {info['spending_range'][0]} - {info['spending_range'][1]}/100")
    
    # ===== STEP 10: Final Output =====
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    # Return results
    return {
        'data': data,
        'scaled_data': scaled_data,
        'pca': pca_2d,
        'pca_df': clustered_pca_df,
        'kmeans': kmeans,
        'clusters': clusters,
        'summary': cluster_summary,
        'ranges': cluster_ranges,
        'interpretations': interpretations
    }


if __name__ == "__main__":
    results = main()
