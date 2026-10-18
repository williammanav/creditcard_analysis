import streamlit as st
# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Credit Card Fraud Detection – Overview",
    page_icon="💳",
    layout="wide"
)

# ===========================
# CUSTOM STYLING
# ===========================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.6px;
    }

    h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 2rem;
    }

    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }

    p, li {
        font-size: 1.05rem !important;
        line-height: 1.7;
        opacity: 0.9;
    }

    .card {
        padding: 1.6rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        margin-bottom: 1.6rem;
    }

    .highlight {
        background: linear-gradient(
            135deg,
            rgba(255, 99, 71, 0.18),
            rgba(255, 165, 0, 0.18)
        );
    }

    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        padding: 1.1rem;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }

    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(
            to right,
            transparent,
            rgba(255,255,255,0.25),
            transparent
        );
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# TITLE
# ===========================
st.title("💳 Credit Card Fraud Detection System")
st.subheader(
    "An end-to-end Machine Learning system for detecting fraudulent credit card transactions"
)
st.image("images/cc1.jpg", width="stretch")

st.markdown("---")

# ===========================
# PROJECT OBJECTIVE
# ===========================
st.markdown(
    """
    <div class="card">
        <h2>🎯 Project Objective</h2>
        <p>
        The goal of this project is to identify <strong>fraudulent credit card transactions</strong>
        in a highly imbalanced, real-world dataset using robust machine learning techniques.
        </p>
        <p>
        Fraud detection is a <strong>cost-sensitive problem</strong>:
        </p>
        <ul>
            <li>❌ Missing fraud (false negative) leads to direct financial loss</li>
            <li>⚠️ Flagging legitimate transactions (false positive) affects customer trust</li>
            <li><strong>Extreme Class Imbalance:</strong> Fraud ≈ 0.17% of all transactions</li>
        </ul>
        <p>
        This system prioritizes <strong>recall, precision-recall metrics, and explainability</strong>
        over raw accuracy.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ===========================
# DATASET SUMMARY
# ===========================
st.markdown("---")
st.header("📊 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Cases", "492")
col3.metric("Fraud Rate", "0.17%")
col4.metric("Features", "30")

st.caption(
    "Dataset contains anonymized, PCA-transformed transaction features from European cardholders."
)

# ===========================
# PROJECT STRUCTURE
# ===========================
st.markdown("---")
st.markdown(
    """
    <div class="card">
        <h2>🧠 Project Structure & Workflow</h2>
        <p>
        The project follows a structured <strong>end-to-end machine learning pipeline</strong>,
        with each stage implemented as a dedicated Streamlit page.
        </p>
        <ol>
            <li><strong>Dataset Explorer</strong> – Raw data inspection and imbalance validation</li>
            <li><strong>EDA & Insights</strong> – Behavioral patterns in fraud vs legitimate transactions</li>
            <li><strong>Feature Engineering</strong> – Log transforms, temporal signals, rolling statistics</li>
            <li><strong>SMOTE</strong> – Handling class imbalance safely on training data</li>
            <li><strong>Model Comparison</strong> – Logistic, RF, XGBoost, LightGBM using PR-AUC</li>
            <li><strong>SHAP Explainability</strong> – Global and local model explanations</li>
            <li><strong>Notebook Report</strong> – Full experimental notebook for transparency</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

# ===========================
# MODELING STRATEGY
# ===========================
st.markdown(
    """
    <div class="card">
        <h2>🤖 Modeling Strategy</h2>
        <ul>
            <li>Baseline model: Logistic Regression</li>
            <li>Ensemble models: Random Forest, XGBoost, LightGBM</li>
            <li>Primary metric: <strong>Precision–Recall AUC</strong></li>
            <li>Threshold tuning to favor fraud recall</li>
        </ul>
        <p>
        Ensemble models are used to capture <strong>non-linear fraud patterns</strong>
        that linear models cannot.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


