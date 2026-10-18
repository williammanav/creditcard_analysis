import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="EDA & Insights",
    page_icon="🔍",
    layout="wide"
)

# ===========================
# Load Data
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df["Hour"] = (df["Time"] / 3600) % 24
    df["ClassLabel"] = df["Class"].map({0: "Legit", 1: "Fraud"})
    return df

df = load_data()

# ===========================
# Sidebar Filters
# ===========================
st.sidebar.title("🔎 Filters")

class_filter = st.sidebar.selectbox(
    "Transaction Type",
    ["All", "Legit Only", "Fraud Only"]
)

amount_range = st.sidebar.slider(
    "Transaction Amount Range",
    float(df["Amount"].min()),
    float(df["Amount"].quantile(0.99)),
    (0.0, float(df["Amount"].quantile(0.95)))
)

hour_range = st.sidebar.slider(
    "Hour of Day",
    0, 23, (0, 23)
)

filtered_df = df.copy()

if class_filter == "Legit Only":
    filtered_df = filtered_df[filtered_df["Class"] == 0]
elif class_filter == "Fraud Only":
    filtered_df = filtered_df[filtered_df["Class"] == 1]

filtered_df = filtered_df[
    (filtered_df["Amount"] >= amount_range[0]) &
    (filtered_df["Amount"] <= amount_range[1]) &
    (filtered_df["Hour"] >= hour_range[0]) &
    (filtered_df["Hour"] <= hour_range[1])
]

COLORS = {"Legit": "#1f77b4", "Fraud": "#d62728"}

# ===========================
# Title
# ===========================
st.title("🔍 Exploratory Data Analysis & Insights")
st.subheader("10 clear visuals explaining fraud behavior")

st.markdown("---")

# ===========================
# Key Metrics
# ===========================
c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{len(filtered_df):,}")
c2.metric("Fraud Cases", int(filtered_df["Class"].sum()))
c3.metric("Fraud Rate", f"{filtered_df['Class'].mean() * 100:.3f}%")

# ===========================
# 1️⃣ Class Distribution
# ===========================
st.header("1️⃣ Transaction Count by Class")

class_counts = filtered_df["ClassLabel"].value_counts().reset_index()
class_counts.columns = ["Transaction Type", "Count"]

fig = px.bar(
    class_counts,
    x="Transaction Type",
    y="Count",
    color="Transaction Type",
    text="Count",
    color_discrete_map=COLORS
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Fraud is extremely rare (<0.2%), making accuracy misleading.")

# ===========================
# 2️⃣ Transaction Share
# ===========================
st.header("2️⃣ Transaction Share (%)")

share = filtered_df["ClassLabel"].value_counts(normalize=True).reset_index()
share.columns = ["Transaction Type", "Share"]

fig = px.bar(
    share,
    x="Transaction Type",
    y="Share",
    text=share["Share"].apply(lambda x: f"{x*100:.2f}%"),
    color="Transaction Type",
    color_discrete_map=COLORS
)
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)
st.caption("Fraud forms a tiny proportion of total transactions.")

# ===========================
# 3️⃣ Amount Distribution
# ===========================
st.header("3️⃣ Transaction Amount Distribution")

fig = px.histogram(
    filtered_df,
    x="Amount",
    nbins=60,
    color="ClassLabel",
    color_discrete_map=COLORS
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Fraud transactions cluster at lower amounts.")

# ===========================
# 4️⃣ Log-Scale Amount Distribution
# ===========================
st.header("4️⃣ Amount Distribution (Log Scale)")

fig = px.histogram(
    filtered_df,
    x="Amount",
    nbins=60,
    log_y=True,
    color="ClassLabel",
    color_discrete_map=COLORS
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Strong right-skew justifies log / robust scaling.")

# ===========================
# 5️⃣ Fraud vs Legit Amount Comparison
# ===========================
st.header("5️⃣ Fraud vs Legit Amount Comparison")

fig = px.box(
    filtered_df,
    x="ClassLabel",
    y="Amount",
    color="ClassLabel",
    color_discrete_map=COLORS
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Fraud shows lower and tighter amount ranges.")

# ===========================
# 6️⃣ Fraud Transactions by Hour
# ===========================
st.header("6️⃣ Fraud Count by Hour")

fraud_counts = (
    filtered_df[filtered_df["Class"] == 1]
    .groupby("Hour")
    .size()
    .reset_index(name="Fraud Count")
)

fig = px.line(
    fraud_counts,
    x="Hour",
    y="Fraud Count",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Fraud activity varies across the day.")

# ===========================
# 7️⃣ Fraud Rate by Hour
# ===========================
st.header("7️⃣ Fraud Probability by Hour")

fraud_rate = (
    filtered_df.groupby("Hour")["Class"]
    .mean()
    .reset_index(name="Fraud Rate")
)

fig = px.bar(
    fraud_rate,
    x="Hour",
    y="Fraud Rate",
    color_discrete_sequence=["#9467bd"]
)
st.plotly_chart(fig, use_container_width=True)
st.caption("This shows risk density, not transaction volume.")

# ===========================
# 8️⃣ Fraud Rate by Amount Bucket
# ===========================
st.header("8️⃣ Fraud Probability by Amount Range")

filtered_df["AmountBin"] = pd.qcut(
    filtered_df["Amount"], q=8, duplicates="drop"
).astype(str)

fraud_rate_amt = (
    filtered_df.groupby("AmountBin")["Class"]
    .mean()
    .reset_index(name="Fraud Rate")
)

fig = px.bar(
    fraud_rate_amt,
    x="AmountBin",
    y="Fraud Rate",
    color="Fraud Rate",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Certain amount ranges carry higher fraud risk.")

# ===========================
# 9️⃣ Cumulative Fraud Capture
# ===========================
st.header("9️⃣ Cumulative Fraud Capture by Amount")

fraud_sorted = (
    filtered_df[filtered_df["Class"] == 1]
    .sort_values("Amount")
    .reset_index(drop=True)
)
fraud_sorted["Cumulative %"] = (
    np.arange(1, len(fraud_sorted) + 1) / len(fraud_sorted)
)

fig = px.line(
    fraud_sorted,
    x="Amount",
    y="Cumulative %",
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Most fraud occurs below certain amount thresholds.")

# ===========================
# 🔟 Volume vs Risk
# ===========================
st.header("🔟 Transaction Volume vs Fraud Risk")

hourly_stats = (
    filtered_df.groupby("Hour")
    .agg(
        Transaction_Count=("Class", "count"),
        Fraud_Rate=("Class", "mean")
    )
    .reset_index()
)

fig = px.scatter(
    hourly_stats,
    x="Transaction_Count",
    y="Fraud_Rate",
    size="Transaction_Count",
    color="Hour"
)
st.plotly_chart(fig, use_container_width=True)
st.caption("High transaction volume does not always imply high fraud risk.")

# ===========================
# Final Insight
# ===========================
st.markdown("---")
st.success(
    """
    **EDA Summary**
    - Fraud is extremely rare but patterned
    - Fraud transactions are usually small
    - Risk varies by time and amount
    - These insights justify SMOTE, threshold tuning, and ensemble models
    """
)
