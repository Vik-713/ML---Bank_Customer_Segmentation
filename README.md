# Case Study 39: Bank Customer Segmentation Using Machine Learning

**Course**: Machine Learning Fundamentals (B.Tech CSE 2024-28, Semester V)  
**Syllabus Alignment**: Modules I to IX (Data Preprocessing, Unsupervised Learning, Dimensionality Reduction, Model Deployment & Evaluation)

This project delivers a complete Machine Learning solution for **Bank Customer Segmentation** based on customer credit card and financial attributes.

---

## 📁 Project Structure

```
ML - Bank Customer Segmentation/
├── Bank_Customer_Segmentation.ipynb # End-to-end Jupyter Notebook pipeline (Data loading, cleaning, K-Means, Hierarchical, PCA & Profiling)
├── Deliverables_Summary.md          # Comprehensive analytical report answering all Section 8 problem statement questions
├── README.md                        # Project documentation and guide
├── .gitignore                       # Git ignore configuration
├── image1.png                       # Case Study 39 Problem Statement (Page 1)
├── image2.png                       # Case Study 39 Problem Statement (Page 2)
├── data/
│   ├── bank_customer_data.csv       # Dataset with 1,500 records & 14 financial features (with missing values)
│   └── bank_customer_data_clustered.csv # Preprocessed dataset with assigned cluster labels
└── plots/
    ├── missing_values_eda.png       # Missing value count per feature
    ├── feature_distributions.png    # Key feature distribution histograms
    ├── correlation_matrix.png       # Correlation matrix heatmap
    ├── elbow_curve.png              # Elbow method (WCSS vs K) plot
    ├── cluster_size_distribution.png # Customer distribution across clusters
    ├── silhouette_scores.png        # Silhouette score evaluation plot (K=2..8)
    ├── dendrogram.png               # Hierarchical clustering dendrogram (Ward linkage)
    ├── pca_clusters_2d.png          # 2D PCA cluster visualization plot
    └── cluster_profiles.png         # Mean attribute profile comparisons per cluster
```

---

## 🚀 How to Run the Notebook

### 1. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### 2. Execute Jupyter Notebook
Open and run all cells in [`Bank_Customer_Segmentation.ipynb`](file:///Users/viketh/Desktop/ML%20-%20Bank%20Customer%20Segmentation/Bank_Customer_Segmentation.ipynb):
```bash
jupyter notebook Bank_Customer_Segmentation.ipynb
```

---

## 📊 Customer Segment Profiling Summary

| Cluster ID | Segment Name | % Base | Key Behavioral Characteristics | Recommended Banking Products |
| :---: | :--- | :---: | :--- | :--- |
| **Cluster #0** | **Conservative Savers / Low Activity** | 35% | Low balances ($\$830$), minimal purchases ($\$251$), low credit limit ($\$3,052$). | High-Yield Savings Account, Basic Zero-Fee Credit Card |
| **Cluster #1** | **Installment Budgeters & Frequent Spenders** | 15% | Moderate balance ($\$1,556$), high installment purchases ($\$1,948$), 28 transactions/yr. | 0% APR Merchant Installment Offers, Cashback Cards, BNPL |
| **Cluster #2** | **High-Balance Power Spenders** | 30% | High balances ($\$5,452$), large one-off purchases ($\$3,253$), high credit limit ($\$12,083$). | Platinum Rewards Cards, Private Wealth Management |
| **Cluster #3** | **Cash-Advance & Liquidity Seekers** | 20% | Heavy cash advances ($\$4,894$), high balance ($\$4,458$), 1.8 active bank loans. | Debt Consolidation Loan, Low-Interest Balance Transfer Card |

---

## 📈 Key Model Results
- **Optimal Cluster Count ($K=4$)**: Confirmed by Elbow Method WCSS knee bend, peak Silhouette score (**0.4846**), and Ward linkage Dendrogram branch truncation.
- **2D PCA Dimensionality Reduction**: PC1 (52.4%) + PC2 (21.6%) capture **74.0% of total variance**.
- **Model Agreement**: K-Means and Hierarchical Clustering achieved an Adjusted Rand Index (ARI) of **1.000**.
- **Cluster Stability**: Average ARI of **1.000** across 5 different random initializations.
