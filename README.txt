# Case Study 39: Bank Customer Segmentation Using Machine Learning

**Course**: Machine Learning Fundamentals (B.Tech CSE 2024-28, Semester V)  
**Syllabus Alignment**: Modules I to IX (Data Preprocessing, Unsupervised Learning, Dimensionality Reduction, Model Deployment & Evaluation)

This project delivers a complete Machine Learning solution for **Bank Customer Segmentation** based on customer credit card and financial attributes, fully deployed as an interactive **Streamlit** web application.

---

## 📁 Project Structure

```
ML - Bank Customer Segmentation/
├── app.py                           # Production-ready Streamlit Web Application (Interactive Dashboard & Real-Time Predictor)
├── Bank_Customer_Segmentation.ipynb # End-to-end Jupyter Notebook pipeline (Data cleaning, K-Means, Hierarchical, PCA & Profiling)
├── Project_Report.txt / .md         # Formal 9-chapter Academic Project Report
├── Problem_Statement_and_Solution_Guide.txt / .md # Comprehensive guide explaining problem statement, solution architecture, and codebase
├── Viva_Preparation_Guide.txt / .md # Oral Exam / Viva Preparation Guide with Plain-English ML Glossary & Q&A
├── README.md / .txt                 # Project documentation and quickstart instructions
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

## 🚀 How to Run

### 1. Environment Setup & Dependencies
```bash
# Navigate to project directory
cd "/Users/viketh/Desktop/ML - Bank Customer Segmentation"

# Activate virtual environment
source venv/bin/activate

# (If needed) Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy streamlit plotly
```

### 2. Launch Streamlit Web Application (Recommended Deployment)
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`** to interactively explore customer data, evaluate cluster models, view segment profiles, and test the real-time customer segment predictor.

### 3. Execute Jupyter Notebook Pipeline
```bash
jupyter notebook Bank_Customer_Segmentation.ipynb
```

---

## 📱 Streamlit Application Features (`app.py`)
- 📌 **Home & Problem Statement**: Executive summary, business context, metric cards, and 5-stage ML workflow.
- 🔍 **EDA & Preprocessing**: Missing value analysis, median imputation justification, feature distributions, and correlation heatmap.
- 📊 **Clustering & Optimal K**: Interactive WCSS Elbow curve, Silhouette score evaluation, Ward linkage Dendrogram, and 2D PCA cluster scatter plot.
- 👥 **Customer Segment Profiles**: Customer proportion pie chart, attribute mean matrix, segment personas, and product recommendations.
- ⚡ **Real-Time Customer Segment Predictor**: Interactive input form with **1-click Persona Preset Loaders** (*Power Spender*, *Saver*, *Cash Advance Seeker*, *Installment Budgeter*) displaying assigned cluster, Softmax confidence score, 2D PCA location, and product recommendations.
- 📚 **Analytical Q&A & Syllabus**: Direct answers to Section 8 problem statement questions and course syllabus mapping matrix (Modules I to IX).

---

## 📊 Customer Segment Profiling Summary

| Cluster ID | Segment Name | % Base | Key Behavioral Characteristics | Recommended Banking Products |
| :---: | :--- | :---: | :--- | :--- |
| **Cluster #0** | **Conservative Savers / Low Activity** | 35% | Low balance ($\$830$), minimal purchases ($\$251$), low credit limit ($\$3,052$). | High-Yield Savings Account, Basic Zero-Fee Credit Card |
| **Cluster #1** | **Installment Budgeters & Frequent Spenders** | 15% | Moderate balance ($\$1,556$), high installment purchases ($\$1,948$), 28 transactions/yr. | 0% APR Merchant Installment Offers, Cashback Cards, BNPL |
| **Cluster #2** | **High-Balance Power Spenders** | 30% | High balances ($\$5,452$), large one-off purchases ($\$3,253$), high credit limit ($\$12,083$). | Platinum Rewards Cards, Private Wealth Management |
| **Cluster #3** | **Cash-Advance & Liquidity Seekers** | 20% | Heavy cash advances ($\$4,894$), high balance ($\$4,458$), 1.8 active bank loans. | Debt Consolidation Loan, Low-Interest Balance Transfer Card |

---

## 📈 Key Model Results
- **Optimal Cluster Count ($K=4$)**: Confirmed by Elbow Method WCSS knee bend (75.6% drop), peak Silhouette score (**0.4846**), and Ward linkage Dendrogram branch truncation.
- **2D PCA Dimensionality Reduction**: PC1 (52.4%) + PC2 (21.6%) capture **74.0% of total variance**.
- **Model Agreement**: K-Means and Hierarchical Clustering achieved an Adjusted Rand Index (ARI) of **1.000**.
- **Cluster Stability**: Average ARI of **1.000** across 5 different random initializations.
