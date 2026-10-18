import os
import pandas as pd
import plotly.express as px
import streamlit as st

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📊",
    layout="wide"
)

# ===========================
# LOAD DATA (SAFE LOADING)
# ===========================
@st.cache_data
def load_data():
    file_path = "creditcard.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["Hour"] = (df["Time"] / 3600) % 24
        df["ClassLabel"] = df["Class"].map({0: "Legit", 1: "Fraud"})
        return df
    return None

df = load_data()

# ===========================
# TITLE & PRE-CHECK
# ===========================
st.title("📊 Dataset Explorer")
st.subheader("Understanding transaction behavior before modeling")

st.markdown("---")

if df is None:
    st.error(
        "⚠️ **Dataset File Not Found**: `creditcard.csv` repository mein nahi mili.\n\n"
        "Kripya dataset file ko project ke root directory mein upload karke push karein."
    )
else:
    st.markdown(
        """
        This page focuses on **understanding the data**, not modeling.
        The insights here directly influence **feature engineering,
        imbalance handling, and metric selection**.
        """
    )

    # ===========================
    # SIDEBAR FILTERS
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

    # Apply filters
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

    # ===========================
    # KEY METRICS
    # ===========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transactions", f"{len(filtered_df):,}")
    col2.metric("Fraud Cases", int(filtered_df["Class"].sum()))
    col3.metric("Fraud Rate", f"{filtered_df['Class'].mean() * 100:.3f}%" if len(filtered_df) > 0 else "0.000%")
    col4.metric("Features", filtered_df.shape[1])

    st.caption("Target variable: Class (Legit = 0, Fraud = 1)")

    # ===========================
    # DATA PREVIEW
    # ===========================
    st.markdown("---")
    st.header("🔍 Data Preview")

    rows = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(filtered_df.head(rows), use_container_width=True)

    # ===========================
    # CLASS DISTRIBUTION
    # ===========================
    st.markdown("---")
    st.header("⚖️ Class Imbalance")

    class_counts = (
        filtered_df["ClassLabel"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = ["Transaction Type", "Count"]

    fig = px.bar(
        class_counts,
        x="Transaction Type",
        y="Count",
        color="Transaction Type",
        color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
        text="Count",
        title="Transaction Class Distribution"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Fraud transactions are extremely rare (<0.2%). "
        "Accuracy alone is misleading for this problem."
    )

    # ===========================
    # AMOUNT DISTRIBUTION
    # ===========================
    st.markdown("---")
    st.header("💰 Transaction Amount Distribution")

    fig = px.histogram(
        filtered_df,
        x="Amount",
        color="ClassLabel",
        nbins=60,
        color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Fraud transactions tend to occur at **lower amounts**, "
        "while legitimate transactions span a wider range."
    )

    # ===========================
    # LOG SCALE AMOUNT
    # ===========================
    fig = px.histogram(
        filtered_df,
        x="Amount",
        color="ClassLabel",
        nbins=60,
        log_y=True,
        color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
        title="Transaction Amount Distribution (Log Scale)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Heavy right skew suggests **log or robust scaling** is required."
    )

    # ===========================
    # FRAUD VS LEGIT COMPARISON
    # ===========================
    st.markdown("---")
    st.header("💳 Fraud vs Legit: Amount Comparison")

    fig = px.box(
        filtered_df,
        x="ClassLabel",
        y="Amount",
        color="ClassLabel",
        color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
        title="Transaction Amount by Class"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Fraud amounts are typically smaller and less variable."
    )

    # ===========================
    # FRAUD RATE BY HOUR
    # ===========================
    st.markdown("---")
    st.header("⏱️ Fraud Risk by Time of Day")

    fraud_rate = (
        filtered_df.groupby("Hour")["Class"]
        .mean()
        .reset_index(name="Fraud Rate")
    )

    fig = px.line(
        fraud_rate,
        x="Hour",
        y="Fraud Rate",
        markers=True,
        title="Fraud Probability by Hour of Day"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This plot shows **risk**, not volume. "
        "Useful for time-based feature engineering."
    )

    # ===========================
    # FRAUD COUNT BY HOUR
    # ===========================
    fraud_counts = (
        filtered_df[filtered_df["Class"] == 1]
        .groupby("Hour")
        .size()
        .reset_index(name="Fraud Count")
    )

    fig = px.bar(
        fraud_counts,
        x="Hour",
        y="Fraud Count",
        title="Fraud Transaction Count by Hour",
        color_discrete_sequence=["#d62728"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # PCA EXPLANATION
    # ===========================
    st.markdown("---")

    if st.button("ℹ️ About PCA Features (V1–V28)"):
        st.info(
            """
            - V1–V28 are **PCA-transformed features**
            - Original transaction details are anonymized
            - PCA removes correlation and protects privacy
            - These features still retain strong fraud signals
            """
        )

    share = (
        filtered_df["ClassLabel"]
        .value_counts(normalize=True)
        .reset_index()
    )
    share.columns = ["Transaction Type", "Share"]

    fig = px.bar(
        share,
        x="Transaction Type",
        y="Share",
        text=share["Share"].apply(lambda x: f"{x*100:.2f}%"),
        color="Transaction Type",
        color_discrete_map={"Legit": "#1f77b4", "Fraud": "#d62728"},
        title="Transaction Share by Class"
    )

    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Fraud makes up a very small fraction of all transactions.")

    # ===========================
    # FINAL INSIGHTS
    # ===========================
    st.markdown("---")

    if st.button("📌 Show Key Dataset Insights"):
        st.success(
            """
            **Key Insights**
            - Fraud is extremely rare → imbalance handling is mandatory
            - Fraud transactions are usually small in amount
            - Amount feature is heavily skewed
            - Fraud likelihood varies by time of day
            - Dataset characteristics justify ensemble models and PR-based metrics
            """
        )