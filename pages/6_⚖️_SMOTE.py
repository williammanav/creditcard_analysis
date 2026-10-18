import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="How SMOTE Works",
    page_icon="⚖️",
    layout="wide"
)

# ===========================
# Load Data
# ===========================
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

# ===========================
# TITLE
# ===========================
st.title("⚖️ How SMOTE Balances Fraud Data")

st.markdown("---")

st.markdown(
    """
    ### 🚨 The Problem
    Fraud detection datasets are **extremely imbalanced**.
    
    Models trained on such data:
    - Learn to predict *everything as legitimate*
    - Achieve high accuracy but **miss fraud**
    """
)

# ===========================
# 1️⃣ ORIGINAL CLASS DISTRIBUTION
# ===========================
st.header("1️⃣ Original Class Imbalance")

class_counts = Counter(df["Class"])
dist_df = pd.DataFrame({
    "Class": ["Legit", "Fraud"],
    "Count": [class_counts[0], class_counts[1]]
})

fig = px.bar(
    dist_df,
    x="Class",
    y="Count",
    color="Class",
    color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
    title="Original Dataset Class Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"""
    Fraud represents only **{df['Class'].mean() * 100:.2f}%** of all transactions.
    
    A model trained directly on this data will **ignore fraud patterns**.
    """
)

# ===========================
# 2️⃣ TRAIN–TEST SPLIT
# ===========================
st.markdown("---")
st.header("2️⃣ Train–Test Split (Important Rule)")

st.write(
    """
     **SMOTE is applied ONLY on the training set**
    
     Applying SMOTE before splitting causes **data leakage**
    """
)

st.code("""
# Initialize SMOTE
smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)

# Apply SMOTE only on training data
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_final,
    y_train_final
)
""")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

before = Counter(y_train)

st.metric("Fraud Samples (Before SMOTE)", before[1])

# ===========================
# 3️⃣ SCALING BEFORE SMOTE
# ===========================
st.markdown("---")
st.header("3️⃣ Scaling Before SMOTE")

st.write(
    """
    SMOTE uses **distance-based interpolation**.
    
    👉 Features must be scaled so that:
    - Large values don’t dominate
    - Distance calculations are meaningful
    """
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ===========================
# 4️⃣ APPLY SMOTE
# ===========================
st.markdown("---")
st.header("4️⃣ Applying SMOTE")

st.write(
    """
    🔁 **SMOTE creates synthetic fraud samples**
    
    It does NOT duplicate points.
    
    Instead, it:
    - Picks a fraud transaction
    - Finds its nearest fraud neighbors
    - Generates a new point **between them**
    """
)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)

after = Counter(y_smote)

col1, col2 = st.columns(2)
col1.metric("Fraud Samples (Before)", before[1])
col2.metric("Fraud Samples (After)", after[1])

# ===========================
# 5️⃣ BEFORE vs AFTER COMPARISON
# ===========================
st.markdown("---")
st.header("5️⃣ Class Balance: Before vs After SMOTE")

compare_df = pd.DataFrame({
    "Stage": ["Before SMOTE", "After SMOTE"],
    "Legit": [before[0], after[0]],
    "Fraud": [before[1], after[1]]
})

compare_df = compare_df.melt(
    id_vars="Stage",
    var_name="Class",
    value_name="Count"
)

fig = px.bar(
    compare_df,
    x="Stage",
    y="Count",
    color="Class",
    barmode="group",
    color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
    title="Effect of SMOTE on Class Balance"
)

st.plotly_chart(fig, use_container_width=True)

st.write("""
Original Training Shape: (227845, 37)
         
SMOTE Training Shape: (454902, 37)
    """)
# ===========================
# 6️⃣ HOW SMOTE GENERATES DATA (PCA VISUAL)
# ===========================
st.markdown("---")
st.header("6️⃣ How SMOTE Generates New Fraud Samples")

st.write(
    """
    Below is a **2D projection** of the training data.
    
    🔵 Blue → Legit  
    🔴 Red → Fraud  
    ✨ New fraud points appear **between existing fraud points**
    """
)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_smote)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Class": np.where(y_smote == 0, "Legit", "Fraud")
})

fig = px.scatter(
    pca_df.sample(5000, random_state=42),
    x="PC1",
    y="PC2",
    color="Class",
    color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
    opacity=0.5,
    title="SMOTE Synthetic Sample Generation (PCA Projection)"
)

st.plotly_chart(fig, use_container_width=True)

# ===========================
# 7️⃣ FINAL TAKEAWAY
# ===========================
st.markdown("---")
st.markdown(
    """
    ###  What SMOTE Achieves
    
    - Balances the dataset **without duplicating fraud samples**
    - Preserves fraud patterns through interpolation
    - Allows models to **learn fraud behavior effectively**
    - Must be applied **only on training data**
    """
)
