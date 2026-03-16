# Mall Customer Segmentation

This project performs **customer segmentation analysis** using machine learning techniques to identify meaningful customer groups in mall retail data.

The analysis combines **Principal Component Analysis (PCA)** and **K-Means clustering** to uncover behavioral patterns based on customer **age, income, and spending habits**. The goal is to understand how different customers behave and how businesses can design targeted marketing strategies based on those segments.


## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results and Insights](#Results-and-Insights)
- [License](#license)

## Project Overview

This analysis uses machine learning techniques to segment mall customers into 3 distinct groups based on:
- **Age**: Customer life stage
- **Annual Income**: Purchasing power
- **Spending Score**: Purchase frequency and volume

The segmentation helps identify high-value customers, budget-conscious shoppers, and other distinct segments for targeted engagement.

## Project Structure

```
mall-customer-segmentation/
├── main.py                      # Main execution script
├── requirements.txt             # Project dependencies
├── LICENSE                      # MIT License
├── README.md                    # This file
├── .gitignore                   # Git ignore patterns
│
├── data/                        # Data directory
│   └── mall_customers.csv       # Customer dataset (input)
│
├── src/                         # Source code modules
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Data preprocessing & scaling
│   ├── pca_analysis.py         # PCA analysis functions
│   ├── clustering.py           # K-Means clustering
│   └── analysis.py             # Cluster analysis & interpretation
│
├── notebooks/                   # Jupyter notebooks
│   └── projectprocess.ipynb    # Exploratory analysis notebook
│
└── output/                      # Generated outputs (gitignored)
    ├── plots/                  # Visualization outputs
    └── results/                # Analysis results
```

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mall-customer-segmentation.git
cd mall-customer-segmentation
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Analysis Pipeline

```bash
python main.py
```

This will:
1. Load and explore the customer data
2. Preprocess and scale features
3. Perform PCA analysis
4. Find optimal number of clusters using elbow method
5. Apply K-Means clustering (k=3)
6. Generate cluster profiles and interpretations
7. Display comprehensive visualization

### Use Individual Modules

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.pca_analysis import apply_pca
from src.clustering import perform_clustering
from src.analysis import get_cluster_summary

# Load and process data
data = load_data('data/mall_customers.csv')
scaled_data = preprocess_data(data)
pca, pca_df = apply_pca(scaled_data)
kmeans, clusters, clustered_df = perform_clustering(pca_df, n_clusters=3)
summary = get_cluster_summary(data, clusters)
print(summary)
```

## Features

### Data Processing
- **Data Loading**: Read CSV files with pandas
- **Preprocessing**: StandardScaler normalization
- **Missing Value Handling**: Automatic detection and reporting

### Dimensionality Reduction
- **PCA Analysis**: Determine optimal principal components
- **Variance Analysis**: Visualize explained variance
- **Component Loadings**: Understand feature contributions

### Clustering
- **Elbow Method**: Find optimal k value
- **K-Means**: Customer segmentation algorithm
- **Visualization**: 2D scatter plots with cluster assignments

### Analysis & Interpretation
- **Cluster Statistics**: Mean values by cluster
- **Gender Distribution**: Gender ratio analysis
- **Customer Profiling**: Automated segment interpretation
- **Summary Reports**: Comprehensive cluster summaries


# Results and Insights

## Analysis Overview

To better understand customer behavior within the mall dataset, a structured data analysis process was performed.

The analysis focused on three key customer attributes:

- **Age**
- **Annual Income**
- **Spending Score**

These variables capture both **purchasing power** and **shopping behavior**, making them useful indicators for identifying different customer segments.

Before performing clustering, the numerical features were **standardized** so that each variable contributes equally to the model. Since the variables operate on different scales (for example, income vs spending score), scaling prevents any single feature from dominating the clustering results.

After scaling the data, **Principal Component Analysis (PCA)** was applied to better understand the relationships between the variables and reduce dimensionality for visualization.

The PCA analysis showed that the **first two principal components explain approximately 77.6% of the total variance**, which is sufficient to represent the structure of the data in two dimensions.

The component loadings revealed an interesting structure in the data:

- **PC1 mainly captures the relationship between Age and Spending Score**, effectively separating younger high-spending customers from older low-spending customers.
- **PC2 primarily represents variation in Annual Income**, distinguishing customers based on purchasing power.

Once the data was projected into this PCA space, **K-Means clustering** was applied to identify natural groupings in the customer population.

The **Elbow Method** was used to determine the optimal number of clusters. The curve showed a clear elbow at **k = 3**, indicating that three clusters provide the most meaningful segmentation without introducing unnecessary complexity.

After clustering, the cluster labels were **mapped back to the original dataset**, allowing the segments to be interpreted using the real customer attributes.

---

## Customer Segments Identified

The analysis revealed **three distinct customer segments**, each representing different spending behaviors and purchasing motivations.

---

### 1. Cautious Affluent Customers

This segment includes customers who have **moderate to relatively high income levels but lower spending scores**.

**Characteristics**

- Average Age: ~49 years  
- Average Income: ~$60k  
- Spending Score: Low  

These customers appear to be **more cautious and value-oriented in their spending behavior**. Even though many of them have sufficient purchasing power, they tend to make purchases carefully and prioritize value when selecting products.

The age range in this cluster spans **from younger adults to older individuals**, indicating that this behavior is driven more by spending mindset than strictly by age.

Customers in this group are likely attracted to:

- Brands offering **good value for money**
- Durable or practical products
- Promotions and discounts

---

### 2. High-Value Customers

This segment represents the **most economically valuable group** in the dataset.

**Characteristics**

- Average Age: ~30 years  
- Average Income: ~$80k  
- Spending Score: High  

These customers combine **strong purchasing power with high spending behavior**. Many appear to be **young professionals or individuals in the early stages of stable careers** who prioritize lifestyle quality and are willing to spend more on products that align with their preferences.

Customers in this segment are more likely to engage with:

- Premium or luxury brands
- Lifestyle products
- High-quality retail experiences

Because of their spending capacity and purchasing behavior, this segment represents an important opportunity for **revenue growth and brand engagement**.

---

### 3. Affordable Luxury Seekers

The third segment represents a group of customers who have **lower income levels but still maintain relatively high spending scores**.

**Characteristics**

- Average Age: ~26 years  
- Average Income: ~$31k  
- Spending Score: High  

These customers appear to be **aspirational buyers** who value lifestyle and trends even though they may have lower purchasing power compared to other groups.

Many customers in this cluster are likely **young individuals or early-career professionals** who are highly engaged with fashion, modern retail experiences, and social trends.

They are often attracted to:

- Trend-driven products
- Mid-range premium brands
- Promotions that make premium experiences more accessible

---

# Marketing and Retail Strategy Implications

The identified segments provide useful insights for designing more targeted marketing and retail strategies within a mall environment.

### Targeting High-Value Customers

Retailers can focus premium offerings toward this segment through:

- Luxury retail stores
- Exclusive product launches
- Loyalty programs
- Personalized promotions

These customers are most responsive to **quality, brand image, and lifestyle-driven marketing**.

---

### Engaging Cautious Affluent Customers

For customers who spend carefully despite having reasonable income levels, effective strategies include:

- Value-focused marketing
- Promotional offers
- Discounts or bundled deals
- Messaging around product durability and long-term value

Retailers emphasizing **quality at competitive prices** are more likely to attract this segment.

---

### Attracting Affordable Luxury Seekers

This segment responds well to **accessible luxury and trend-focused retail experiences**.

Effective strategies may include:

- Mid-tier premium brands
- Seasonal promotions
- Trend-driven fashion marketing
- Social media campaigns highlighting lifestyle appeal

These customers are often motivated by **style, identity, and modern consumer trends**.

---

# Retail Layout Insights

Customer segmentation can also help guide **store placement and promotional strategy within the mall**.

For example:

- **Luxury brand zones** can target high-value customers.
- **Value-focused retail areas** can attract cautious shoppers.
- **Fashion and lifestyle-focused stores** can appeal to younger aspirational customers.

Digital displays, in-mall advertising, and promotional campaigns can also be customized based on the motivations of each segment.

---

# Key Takeaways

- Customer behavior in the dataset is primarily driven by **income level and spending habits**.
- PCA revealed that **age and spending behavior are strongly related**, while income forms a separate axis of variation.
- Three meaningful customer segments were identified using K-Means clustering.
- Each segment represents **distinct motivations and purchasing patterns**.
- Retailers and mall management can use these insights to design **more targeted marketing strategies and store experiences**.

---

# Business Insights

From a business perspective, segmentation provides several actionable insights:

- **High-value customers** represent the most important segment for long-term revenue growth.
- **Affordable luxury seekers** represent a strong opportunity for trend-focused retail and lifestyle brands.
- **Cautious affluent customers** may increase spending when presented with strong value propositions.

Understanding these customer segments enables businesses to move from **generic marketing approaches to more personalized and targeted retail strategies**.

## Dataset

**mall_customers.csv** contains:
- **CustomerID**: Unique identifier (200 customers)
- **Gender**: Male/Female distribution
- **Age**: Range 19-45 years
- **Annual Income**: $15k-$137k range
- **Spending Score**: 1-100 scale of purchase behavior

## Technologies Used

- **Python 3.7+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization

## Module Overview

### `src/data_loader.py`
- `load_data()`: Load CSV data
- `get_data_info()`: Display dataset statistics

### `src/preprocessing.py`
- `preprocess_data()`: Scale features using StandardScaler

### `src/pca_analysis.py`
- `perform_pca_analysis()`: Full PCA analysis
- `apply_pca()`: Apply PCA with n components
- `get_pca_loadings()`: Extract component loadings
- `plot_explained_variance()`: Visualize variance

### `src/clustering.py`
- `find_optimal_clusters()`: Elbow method analysis
- `perform_clustering()`: K-Means clustering
- `plot_clusters()`: Visualize segments
- `plot_elbow_curve()`: Plot elbow curve

### `src/analysis.py`
- `get_cluster_statistics()`: Numerical statistics
- `get_cluster_gender_distribution()`: Gender analysis
- `get_cluster_summary()`: Comprehensive summary
- `interpret_clusters()`: Generate interpretations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Talha Amin

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset inspired by real-world retail customer behavior
- Techniques based on classic machine learning practices
- Visualization inspired by professional data science standards

