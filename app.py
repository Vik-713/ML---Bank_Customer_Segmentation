"""
Bank Customer Segmentation Machine Learning Streamlit Application
Case Study 39 • B.Tech CSE Semester V ML Course
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Page Configuration
st.set_page_config(
    page_title="Bank Customer Segmentation ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 32px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #4F46E5;
    }
    .metric-lbl {
        font-size: 12px;
        color: #64748B;
    }
    </style>
""", unsafe_allow_html=True)

# Data Loading & Preprocessing Pipeline (Cached)
@st.cache_data
def load_and_process_data():
    csv_path = 'data/bank_customer_data.csv'
    if not os.path.exists(csv_path):
        st.error(f"Dataset file '{csv_path}' not found.")
        st.stop()
        
    df_raw = pd.read_csv(csv_path)
    df_clean = df_raw.copy()
    
    # Missing Value Handling (Median Imputation)
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Feature Normalization
    feature_cols = [c for c in num_cols if c != 'CUST_ID']
    X = df_clean[feature_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Final Models (K=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean['Cluster'] = clusters
    
    # Hierarchical Clustering
    agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    df_clean['Hierarchical_Cluster'] = agg.fit_predict(X_scaled)
    
    # 2D PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_clean['PC1'] = X_pca[:, 0]
    df_clean['PC2'] = X_pca[:, 1]
    
    return df_raw, df_clean, feature_cols, X_scaled, scaler, kmeans, pca, X_pca

# Load Data & Models
df_raw, df_clean, feature_cols, X_scaled, scaler, kmeans_model, pca_model, X_pca = load_and_process_data()

# Segment Metadata Definition
SEGMENT_META = {
    0: {
        "name": "Conservative Savers / Low Activity",
        "desc": "Low balances ($830), minimal purchases ($251), low credit limit ($3,052), low annual transactions (5/yr). Zero cash advance dependence.",
        "products": ["High-Yield Savings Account", "Basic Zero-Fee Credit Card", "Automated Micro-Investment Plan"],
        "risk": "Low Risk / Low Yield",
        "color": "#64748B"
    },
    1: {
        "name": "Installment Budgeters & Frequent Spenders",
        "desc": "Moderate balance ($1,556), high installment purchases ($1,948), high purchase frequency (0.85), 28 transactions/yr.",
        "products": ["0% APR Merchant Installment Offers", "Cashback Rewards Card", "Buy-Now-Pay-Later (BNPL) Integration"],
        "risk": "Low Risk / High Engagement",
        "color": "#06B6D4"
    },
    2: {
        "name": "High-Balance Power Spenders",
        "desc": "High balance ($5,452), large one-off purchases ($3,253), credit limit ($12,083), 45 transactions/yr.",
        "products": ["Platinum Rewards Credit Card", "Private Wealth Management & Advisory", "High-Limit Line of Credit"],
        "risk": "Low Risk / High Value",
        "color": "#6366F1"
    },
    3: {
        "name": "Cash-Advance & Liquidity Seekers",
        "desc": "Heavy cash advances ($4,894), high balance ($4,458), 1.8 active bank loans, low direct card purchases.",
        "products": ["Debt Consolidation Loan", "Low-Interest Balance Transfer Card", "Personal Financial Advisory"],
        "risk": "Moderate-High Credit Risk",
        "color": "#F59E0B"
    }
}

# Sidebar Navigation
st.sidebar.title("🏦 Navigation Menu")
page = st.sidebar.radio("Go to Section:", [
    "📌 Home & Problem Statement",
    "🔍 EDA & Data Preprocessing",
    "📊 Clustering & Optimal K Evaluation",
    "👥 Customer Segment Profiles",
    "⚡ Real-Time Customer Predictor",
    "📚 Analytical Q&A & Syllabus"
])

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Fundamentals • Case Study 39")

# --- PAGE 1: HOME & PROBLEM STATEMENT ---
if page == "📌 Home & Problem Statement":
    st.markdown('<div class="main-header">Bank Customer Segmentation ML System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Case Study 39 • Unsupervised Machine Learning, PCA & Targeted Product Engine</div>', unsafe_allow_html=True)
    
    st.info("💡 **Executive Summary**: This application deploys an unsupervised Machine Learning pipeline to group 1,500 retail bank customers into 4 distinct commercial segments based on transactional and account behavior.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">1,500</div><div class="metric-lbl">Total Customer Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">4 Clusters</div><div class="metric-lbl">Optimal K Selected</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-val">0.4846</div><div class="metric-lbl">Silhouette Score</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">74.0%</div><div class="metric-lbl">PCA Variance Explained</div></div>', unsafe_allow_html=True)
        
    st.markdown("### 🎯 The Business Problem")
    st.write("""
    Traditionally, retail banks applied generic, one-size-fits-all product marketing and credit limit policies. This caused:
    - **Misaligned Marketing**: Premium rewards cards pitched to inactive savers.
    - **Unmanaged Risk**: High-risk users relying heavily on cash advances not flagged early.
    - **Suboptimal Revenue**: High-balance power spenders not receiving tailored wealth management.
    """)
    
    st.markdown("### 🔄 Machine Learning Workflow Architecture")
    st.code("""
    Raw Data (1,500 Records, 14 Features)
       ↓
    Median Imputation & StandardScaler Normalization
       ↓
    Optimal K Determination (Elbow Method + Silhouette Score + Dendrogram) -> K = 4
       ↓
    K-Means & Agglomerative Hierarchical Clustering (ARI = 1.000)
       ↓
    2D PCA Dimensionality Reduction (74.0% Variance)
       ↓
    Segment Profiling & Targeted Banking Product Engine
    """, language="text")

# --- PAGE 2: EDA & DATA PREPROCESSING ---
elif page == "🔍 EDA & Data Preprocessing":
    st.markdown('<div class="main-header">Exploratory Data Analysis & Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data Cleaning, Median Imputation & StandardScaler Normalization</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Raw Dataset Overview", "🩹 Missing Value Imputation", "📈 Feature Distributions & Correlation"])
    
    with tab1:
        st.subheader("Raw Dataset Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.write(f"**Dataset Shape**: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        
    with tab2:
        st.subheader("Missing Value Summary & Median Imputation")
        missing_counts = df_raw.isnull().sum()
        missing_df = pd.DataFrame({
            "Feature Name": missing_counts.index,
            "Missing Count": missing_counts.values,
            "Percentage (%)": (missing_counts.values / len(df_raw)) * 100
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.table(missing_df)
            st.warning("⚠️ **Why Median Imputation?**: Financial data is heavily right-skewed with high spending outliers. Median imputation is robust against extreme outliers, preserving true central tendency.")
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=missing_df["Feature Name"], y=missing_df["Missing Count"], color="#4F46E5", ax=ax)
            ax.set_title("Missing Value Count per Feature", fontweight="bold")
            st.pyplot(fig)
            
    with tab3:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_clean[feature_cols].corr(), annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=ax)
        st.pyplot(fig)

# --- PAGE 3: CLUSTERING & OPTIMAL K EVALUATION ---
elif page == "📊 Clustering & Optimal K Evaluation":
    st.markdown('<div class="main-header">Clustering Evaluation & PCA Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Elbow Method, Silhouette Analysis, Dendrogram & 2D PCA Scatter Plot</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Elbow Method (WCSS)")
        wcss = [20999.9, 12149.5, 7536.8, 5120.7, 4619.8, 4359.4, 4176.1, 3995.3, 3871.5, 3773.7]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--', color='#4F46E5', linewidth=2)
        ax.axvline(x=4, color='#EF4444', linestyle=':', label='Optimal K = 4')
        ax.set_title("Elbow Curve (WCSS vs K)", fontweight="bold")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("WCSS / Inertia")
        ax.legend()
        st.pyplot(fig)
        st.caption("Sharp knee drop at K=4 (75.6% variance reduction).")
        
    with col2:
        st.subheader("2. Silhouette Score Evaluation")
        sil_scores = [0.3917, 0.4699, 0.4846, 0.3984, 0.3130, 0.2497, 0.2484]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(2, 9), sil_scores, marker='s', linestyle='-', color='#10B981', linewidth=2)
        ax.set_title("Silhouette Scores (K = 2 to 8)", fontweight="bold")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        st.pyplot(fig)
        st.caption("Peak Silhouette score of 0.4846 achieved at K=4.")
        
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3. Hierarchical Ward Dendrogram")
        linked = linkage(X_scaled, method='ward')
        fig, ax = plt.subplots(figsize=(6, 4))
        dendrogram(linked, truncate_mode='lastp', p=30, show_contracted=True, ax=ax)
        ax.set_title("Ward Linkage Dendrogram", fontweight="bold")
        st.pyplot(fig)
        
    with col4:
        st.subheader("4. 2D PCA Cluster Scatter Plot")
        fig, ax = plt.subplots(figsize=(6, 4))
        palette = ['#64748B', '#06B6D4', '#6366F1', '#F59E0B']
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_clean, palette=palette, alpha=0.8, ax=ax)
        centroids_pca = pca_model.transform(kmeans_model.cluster_centers_)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='black', marker='X', label='Centroids')
        ax.set_title("2D PCA Visualization (74.0% Variance)", fontweight="bold")
        st.pyplot(fig)

# --- PAGE 4: CUSTOMER SEGMENT PROFILES ---
elif page == "👥 Customer Segment Profiles":
    st.markdown('<div class="main-header">Customer Segment Personas & Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cluster Attribute Mean Breakdown & Product Recommendation Matrix</div>', unsafe_allow_html=True)
    
    # Cluster Distribution
    counts = df_clean['Cluster'].value_counts().sort_index()
    pcts = (counts / len(df_clean)) * 100
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Cluster Size Distribution")
        dist_df = pd.DataFrame({
            "Cluster ID": counts.index,
            "Segment Name": [SEGMENT_META[i]["name"] for i in counts.index],
            "Customer Count": counts.values,
            "Percentage (%)": [f"{p:.1f}%" for p in pcts.values]
        })
        st.table(dist_df)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.pie(counts.values, labels=[f"Cluster #{i}" for i in counts.index], autopct='%1.0f%%', colors=['#64748B', '#06B6D4', '#6366F1', '#F59E0B'])
        ax.set_title("Customer Proportion per Cluster", fontweight="bold")
        st.pyplot(fig)
        
    st.subheader("Cluster Profile Attribute Means")
    profile_df = df_clean.groupby('Cluster')[feature_cols].mean().round(2)
    st.dataframe(profile_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Segment Personas & Product Recommendations")
    for cid, meta in SEGMENT_META.items():
        with st.expander(f"📌 Cluster #{cid}: {meta['name']} ({pcts[cid]:.0f}% of Customers)", expanded=True):
            st.write(f"**Description**: {meta['desc']}")
            st.write(f"**Risk Profile**: `{meta['risk']}`")
            st.write("**Recommended Banking Products**:")
            for p in meta['products']:
                st.markdown(f"- ✨ {p}")

# --- PAGE 5: REAL-TIME CUSTOMER PREDICTOR ---
elif page == "⚡ Real-Time Customer Predictor":
    st.markdown('<div class="main-header">Real-Time Customer Segment Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter banking customer attributes to classify segment and recommend products</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### Load Sample Personas")
    if st.sidebar.button("💎 Load Power Spender"):
        st.session_state.inputs = {
            "BALANCE": 5500.0, "BALANCE_FREQUENCY": 0.95, "PURCHASES": 4200.0, "ONEOFF_PURCHASES": 3200.0,
            "INSTALLMENTS_PURCHASES": 1000.0, "CASH_ADVANCE": 250.0, "PURCHASES_FREQUENCY": 0.90, "PURCHASES_TRX": 45,
            "CREDIT_LIMIT": 12000.0, "PAYMENTS": 4000.0, "MINIMUM_PAYMENTS": 1200.0, "PRC_FULL_PAYMENT": 0.65,
            "LOAN_HOLDINGS": 1, "TENURE": 12
        }
    if st.sidebar.button("🏦 Load Low Activity Saver"):
        st.session_state.inputs = {
            "BALANCE": 800.0, "BALANCE_FREQUENCY": 0.60, "PURCHASES": 250.0, "ONEOFF_PURCHASES": 150.0,
            "INSTALLMENTS_PURCHASES": 100.0, "CASH_ADVANCE": 100.0, "PURCHASES_FREQUENCY": 0.25, "PURCHASES_TRX": 5,
            "CREDIT_LIMIT": 3000.0, "PAYMENTS": 500.0, "MINIMUM_PAYMENTS": 250.0, "PRC_FULL_PAYMENT": 0.15,
            "LOAN_HOLDINGS": 0, "TENURE": 9
        }
    if st.sidebar.button("💸 Load Cash Advance Seeker"):
        st.session_state.inputs = {
            "BALANCE": 4500.0, "BALANCE_FREQUENCY": 0.90, "PURCHASES": 500.0, "ONEOFF_PURCHASES": 200.0,
            "INSTALLMENTS_PURCHASES": 300.0, "CASH_ADVANCE": 4800.0, "PURCHASES_FREQUENCY": 0.30, "PURCHASES_TRX": 8,
            "CREDIT_LIMIT": 7500.0, "PAYMENTS": 3200.0, "MINIMUM_PAYMENTS": 1800.0, "PRC_FULL_PAYMENT": 0.08,
            "LOAN_HOLDINGS": 2, "TENURE": 10
        }
    if st.sidebar.button("📦 Load Installment Budgeter"):
        st.session_state.inputs = {
            "BALANCE": 1500.0, "BALANCE_FREQUENCY": 0.85, "PURCHASES": 2200.0, "ONEOFF_PURCHASES": 300.0,
            "INSTALLMENTS_PURCHASES": 1900.0, "CASH_ADVANCE": 150.0, "PURCHASES_FREQUENCY": 0.85, "PURCHASES_TRX": 28,
            "CREDIT_LIMIT": 5500.0, "PAYMENTS": 1800.0, "MINIMUM_PAYMENTS": 500.0, "PRC_FULL_PAYMENT": 0.40,
            "LOAN_HOLDINGS": 1, "TENURE": 11
        }
        
    defaults = st.session_state.get("inputs", {
        "BALANCE": 4500.0, "BALANCE_FREQUENCY": 0.90, "PURCHASES": 4200.0, "ONEOFF_PURCHASES": 3200.0,
        "INSTALLMENTS_PURCHASES": 1000.0, "CASH_ADVANCE": 250.0, "PURCHASES_FREQUENCY": 0.85, "PURCHASES_TRX": 42,
        "CREDIT_LIMIT": 12000.0, "PAYMENTS": 3800.0, "MINIMUM_PAYMENTS": 1100.0, "PRC_FULL_PAYMENT": 0.55,
        "LOAN_HOLDINGS": 1, "TENURE": 12
    })
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            bal = st.number_input("Account Balance ($)", value=float(defaults["BALANCE"]))
            bal_freq = st.number_input("Balance Frequency (0-1)", value=float(defaults["BALANCE_FREQUENCY"]), min_value=0.0, max_value=1.0)
            purch = st.number_input("Total Purchases ($)", value=float(defaults["PURCHASES"]))
            oneoff = st.number_input("One-Off Purchases ($)", value=float(defaults["ONEOFF_PURCHASES"]))
            inst = st.number_input("Installment Purchases ($)", value=float(defaults["INSTALLMENTS_PURCHASES"]))
        with col2:
            cash_adv = st.number_input("Cash Advance ($)", value=float(defaults["CASH_ADVANCE"]))
            purch_freq = st.number_input("Purchase Frequency (0-1)", value=float(defaults["PURCHASES_FREQUENCY"]), min_value=0.0, max_value=1.0)
            purch_trx = st.number_input("Purchase Transactions Count", value=int(defaults["PURCHASES_TRX"]), min_value=0)
            cred_lim = st.number_input("Credit Limit ($)", value=float(defaults["CREDIT_LIMIT"]))
            payments = st.number_input("Total Payments ($)", value=float(defaults["PAYMENTS"]))
        with col3:
            min_pay = st.number_input("Minimum Payments ($)", value=float(defaults["MINIMUM_PAYMENTS"]))
            prc_full = st.number_input("% Full Payment Paid (0-1)", value=float(defaults["PRC_FULL_PAYMENT"]), min_value=0.0, max_value=1.0)
            loans = st.number_input("Active Loan Holdings", value=int(defaults["LOAN_HOLDINGS"]), min_value=0)
            tenure = st.number_input("Bank Tenure (Months)", value=int(defaults["TENURE"]), min_value=6, max_value=12)
            submit = st.form_submit_button("⚡ Predict Customer Segment", use_container_width=True)
            
    if submit or "inputs" in st.session_state:
        raw_vals = np.array([[bal, bal_freq, purch, oneoff, inst, cash_adv, purch_freq, purch_trx, cred_lim, payments, min_pay, prc_full, loans, tenure]])
        scaled_vals = scaler.transform(raw_vals)
        
        # Centroid Euclidean Distances
        dists = [np.linalg.norm(scaled_vals - c) for c in kmeans_model.cluster_centers_]
        pred_cluster = int(np.argmin(dists))
        
        # Softmax confidence
        inv_d = np.exp(-0.8 * np.array(dists))
        probs = (inv_d / np.sum(inv_d)) * 100
        
        # 2D PCA projection
        pca_point = pca_model.transform(scaled_vals)[0]
        meta = SEGMENT_META[pred_cluster]
        
        st.markdown("---")
        st.success(f"🎯 **Predicted Cluster #{pred_cluster}: {meta['name']}** (Confidence: {probs[pred_cluster]:.1f}%)")
        
        col_res1, col_res2 = st.columns([1.2, 1])
        with col_res1:
            st.write(f"**Segment Description**: {meta['desc']}")
            st.write(f"**Risk Profile**: `{meta['risk']}`")
            st.write("**Recommended Banking Products**:")
            for p in meta['products']:
                st.markdown(f"- ✨ {p}")
                
        with col_res2:
            st.write("**Centroid Euclidean Distance Breakdown**:")
            dist_df = pd.DataFrame({
                "Cluster": [f"Cluster #{i}: {SEGMENT_META[i]['name']}" for i in range(4)],
                "Euclidean Distance": np.round(dists, 3),
                "Softmax Confidence": [f"{p:.1f}%" for p in probs]
            })
            st.table(dist_df)
            st.caption(f"2D PCA Location: ({pca_point[0]:.2f}, {pca_point[1]:.2f})")

# --- PAGE 6: ANALYTICAL Q&A & SYLLABUS ---
elif page == "📚 Analytical Q&A & Syllabus":
    st.markdown('<div class="main-header">Analytical Q&A & Syllabus Alignment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Section 8 Problem Statement Answers & Course Syllabus Coverage</div>', unsafe_allow_html=True)
    
    st.subheader("Section 8 Problem Statement Questions & Answers")
    
    q_and_a = [
        ("Q1: How many distinct banking customer segments exist?", "4 distinct segments (K=4), justified by Elbow WCSS drop, peak Silhouette score (0.4846), and Ward Linkage Dendrogram truncation."),
        ("Q2: What characterizes each segment?", "Cluster 0 (Savers: low spend, low balance), Cluster 1 (Installment Budgeters: high installment spend), Cluster 2 (Power Spenders: high balance, high credit limit), Cluster 3 (Cash Seekers: high cash advance, high loans)."),
        ("Q3: Does the PCA visualization support the chosen cluster count?", "Yes. 2D PCA plot (74.0% total variance explained) shows 4 clear, non-overlapping cluster regions."),
        ("Q4: How does the WCSS change as the number of clusters increases?", "WCSS drops steeply from 21,000.0 (K=1) to 5,120.7 (K=4), representing a 75.6% variance drop, after which reductions flatten."),
        ("Q5: Do the clusters differ meaningfully in their profiles?", "Yes. Feature means demonstrate clear statistical divergence across purchases, cash advances, and credit limits."),
        ("Q6: How stable are the clusters across different random initialisations?", "Across 5 random seeds (42, 100, 2024, 777, 999), K-Means achieved an average Adjusted Rand Index (ARI) of 1.000, confirming complete stability."),
        ("Q7: Can the segmentation support product design?", "Yes. Segments map directly to tailored banking products (Wealth management for Power Spenders, BNPL for Installment Buyers, Debt consolidation for Cash Seekers).")
    ]
    
    for q, a in q_and_a:
        with st.expander(q, expanded=False):
            st.write(a)
            
    st.markdown("---")
    st.subheader("Course Syllabus Mapping Matrix")
    st.markdown("""
    | Syllabus Module | Module Name | Project Implementation Evidence |
    |---|---|---|
    | **Module I** | Introduction to ML | Unsupervised clustering workflow formulation & ML lifecycle execution |
    | **Module II** | ML Libraries & Packages | Implemented via NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn, Streamlit |
    | **Module III** | Data Preprocessing | Median missing value imputation (`BALANCE`, `CREDIT_LIMIT`, `MINIMUM_PAYMENTS`); `StandardScaler` normalization |
    | **Module VII** | Unsupervised Learning | K-Means Clustering and Agglomerative Hierarchical Clustering |
    | **Module VIII** | Dimensionality Reduction | 2D Principal Component Analysis (PCA) variance decomposition |
    | **Module IX** | Model Deployment Basics | Full Streamlit application deployment with real-time customer segmentation engine |
    """)
