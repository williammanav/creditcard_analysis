import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🧠",
    layout="wide"
)

# ===========================
# Load Data
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv").sort_values("Time")
    df["Hour"] = (df["Time"] / 3600) % 24
    df["ClassLabel"] = df["Class"].map({0: "Legit", 1: "Fraud"})
    return df

df = load_data()

# ===========================
# SIDEBAR CONTROLS
# ===========================
st.sidebar.title("⚙️ Feature Controls")

rolling_window = st.sidebar.slider(
    "Rolling Window Size",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

sample_size = st.sidebar.select_slider(
    "Sample Size (visualization)",
    options=[1_000, 5_000, 10_000, 25_000, len(df)],
    value=10_000
)

show_pca = st.sidebar.checkbox("Show PCA Aggregation", value=True)

viz_df = df.head(sample_size).copy()

# ===========================
# TITLE
# ===========================
st.title("🧠 Feature Engineering")
st.subheader("Transforming raw transactions into behavioral signals")

st.markdown(
    """
    Feature engineering converts **raw transactions** into **behavioral indicators**
    that machine learning models can effectively learn from.
    """
)

st.markdown("---")

# ===========================
# RAW FEATURES
# ===========================
st.header("🔹 Raw Input Features")

st.dataframe(
    viz_df[["Time", "Amount", "Class"]].head(10),
    use_container_width=True
)

st.caption(
    "The dataset contains PCA features (V1–V28), raw numerical fields, and a binary fraud label."
)

# ===========================
# AMOUNT-BASED FEATURES
# ===========================
st.markdown("---")
st.header("💰 Amount-Based Feature Engineering")

viz_df["log_amount"] = np.log1p(viz_df["Amount"])

fig = px.histogram(
    viz_df,
    x="Amount",
    nbins=60,
    title="Raw Transaction Amount Distribution"
)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    viz_df,
    x="log_amount",
    nbins=60,
    title="Log-Transformed Transaction Amount"
)
st.plotly_chart(fig, use_container_width=True)

st.success(
    "Log transformation reduces skewness and stabilizes model learning."
)

# ===========================
# TIME-BASED FEATURES
# ===========================
st.markdown("---")
st.header("⏱️ Time-Based Feature Engineering")

viz_df["time_diff"] = viz_df["Time"].diff().fillna(0)

fig = px.histogram(
    viz_df,
    x="time_diff",
    nbins=60,
    title="Time Difference Between Consecutive Transactions"
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Rapid transaction bursts often indicate automated or fraudulent behavior."
)

# ===========================
# ROLLING BEHAVIORAL FEATURES
# ===========================
st.markdown("---")
st.header("📈 Rolling Behavioral Features")

viz_df["rolling_amount_mean"] = (
    viz_df["Amount"].rolling(window=rolling_window, min_periods=1).mean()
)

viz_df["rolling_amount_std"] = (
    viz_df["Amount"].rolling(window=rolling_window, min_periods=1).std().fillna(0)
)

fig = px.line(
    viz_df.head(500),
    y="rolling_amount_mean",
    title=f"Rolling Mean of Transaction Amount (Window = {rolling_window})"
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Rolling statistics capture short-term spending trends and sudden changes."
)

# ===========================
# PCA FEATURE AGGREGATION
# ===========================
if show_pca:
    st.markdown("---")
    st.header("🧩 PCA Feature Aggregation")

    pca_features = [c for c in df.columns if c.startswith("V")]

    viz_df["pca_sum"] = viz_df[pca_features].sum(axis=1)
    viz_df["pca_std"] = viz_df[pca_features].std(axis=1)

    fig = px.histogram(
        viz_df,
        x="pca_sum",
        color="ClassLabel",
        nbins=60,
        color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
        title="Aggregated PCA Signal Strength"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Aggregated PCA features act as a global anomaly signal across dimensions."
    )

# ===========================
# FEATURE SUMMARY TABLE
# ===========================
st.markdown("---")
st.header("🧠 Engineered Feature Summary")

feature_table = pd.DataFrame({
    "Feature": [
        "log_amount",
        "Hour",
        "time_diff",
        f"rolling_amount_mean (window={rolling_window})",
        f"rolling_amount_std (window={rolling_window})",
        "pca_sum",
        "pca_std"
    ],
    "Purpose": [
        "Reduce skew in transaction amount",
        "Capture daily fraud cycles",
        "Detect transaction bursts",
        "Short-term spending behavior",
        "Spending volatility",
        "Overall anomaly magnitude",
        "Anomaly dispersion"
    ]
})

st.dataframe(feature_table, use_container_width=True)

# ===========================
# FINAL OUTCOME
# ===========================
st.success(
    """
    **Outcome of Feature Engineering**
    - Raw numerical data transformed into behavioral signals
    - Features capture magnitude, velocity, and volatility of transactions
    - These engineered features significantly improve ensemble model performance
    """
)
