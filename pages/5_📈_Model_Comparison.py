import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import *


st.set_page_config(
    page_title="Model Comparison",
    page_icon="📈",
    layout="wide"
)

# ===========================
# Load precomputed results
# ===========================
@st.cache_data
def load_results():
    return np.load("results/model_outputs.npy", allow_pickle=True).item()

models = load_results()

# ===========================
# SIDEBAR CONTROLS
# ===========================
st.sidebar.title("⚙️ Model Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

# ===========================
# Extract model data
# ===========================
y_true = models[model_name]["y_true"]
y_proba = models[model_name]["y_proba"]
y_pred = (y_proba >= threshold).astype(int)

# ===========================
# Title
# ===========================
st.title("📈 Model Comparison Dashboard")
st.subheader("Evaluating fraud detection models under class imbalance")

st.markdown("---")

# ===========================
# Metrics
# ===========================
precision, recall, _ = precision_recall_curve(y_true, y_proba)
pr_auc = auc(recall, precision)

c1, c2, c3 = st.columns(3)
c1.metric("Selected Model", model_name)
c2.metric("PR-AUC", f"{pr_auc:.3f}")
c3.metric("Threshold", threshold)

# ===========================
# Confusion Matrix
# ===========================
st.markdown("---")
st.header("🧮 Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Legit", "Actual Fraud"],
    columns=["Pred Legit", "Pred Fraud"]
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Confusion Matrix"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Adjusting the threshold shifts the balance between false positives and false negatives."
)

# ===========================
# Precision–Recall Curve
# ===========================
st.markdown("---")
st.header("📉 Precision–Recall Curve")

pr_df = pd.DataFrame({
    "Recall": recall,
    "Precision": precision
})

fig = px.line(
    pr_df,
    x="Recall",
    y="Precision",
    title=f"Precision–Recall Curve (PR-AUC = {pr_auc:.3f})"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "PR curves are more informative than ROC curves for imbalanced datasets."
)

# ===========================
# Classification Report
# ===========================
st.markdown("---")
st.header("📄 Classification Report")

report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df.round(3), use_container_width=True)


# ===========================
# Cross-Model PR-AUC Comparison
# ===========================
st.markdown("---")
st.header("🏆 Cross-Model Performance Comparison")

scores = []

for name, data in models.items():
    p, r, _ = precision_recall_curve(data["y_true"], data["y_proba"])
    scores.append({
        "Model": name,
        "PR-AUC": auc(r, p)
    })

score_df = pd.DataFrame(scores)

fig = px.bar(
    score_df,
    x="Model",
    y="PR-AUC",
    color="Model",
    title="PR-AUC Comparison Across Models"
)

st.plotly_chart(fig, use_container_width=True)

thresholds = np.linspace(0.01, 0.99, 50)
recalls = []

for t in thresholds:
    y_tmp = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_true, y_tmp)
    recall_val = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-9)
    recalls.append(recall_val)

fig = px.line(
    x=thresholds,
    y=recalls,
    labels={"x": "Threshold", "y": "Recall"},
    title="Recall vs Decision Threshold"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Lower thresholds increase fraud capture but raise false positives.")

precisions = []

for t in thresholds:
    y_tmp = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_true, y_tmp)
    precision_val = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-9)
    precisions.append(precision_val)

fig = px.line(
    x=thresholds,
    y=precisions,
    labels={"x": "Threshold", "y": "Precision"},
    title="Precision vs Decision Threshold"
)

st.plotly_chart(fig, use_container_width=True)
false_positives = []

for t in thresholds:
    y_tmp = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_true, y_tmp)
    false_positives.append(cm[0,1])

fig = px.line(
    x=thresholds,
    y=false_positives,
    labels={"x": "Threshold", "y": "False Positives"},
    title="False Positives vs Threshold"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Higher thresholds reduce customer friction.")

# ===========================
# Key Insights
# ===========================
st.markdown("---")
st.success(
    """
    **Key Insights**
    - PR-AUC is the correct metric for extreme class imbalance
    - Ensemble models outperform logistic regression
    - Threshold tuning allows control over recall vs precision
    - LightGBM and XGBoost provide the best overall fraud detection performance
    """
)

st.header("🚨 The Core Problem: Extreme Class Imbalance")

st.markdown(
    """
    In fraud detection:
    
    - **Legitimate transactions ≫ Fraud transactions**
    - Fraud is often **< 0.2%** of the data
    - A model can achieve **99.8% accuracy** by predicting *everything as legitimate*
    
    👉 **Accuracy becomes meaningless**
    """
)

st.info(
    "We need a metric that focuses on **fraud detection quality**, not overall correctness."
)

# ===========================
# SECTION 2: WHAT MATTERS IN FRAUD?
# ===========================
st.header("🔍 What Actually Matters in Fraud Detection?")

col1, col2 = st.columns(2)

with col1:
    st.success(
        """
        **Recall (Fraud Capture)**
        - How many frauds did we catch?
        - Missing fraud is costly
        """
    )

with col2:
    st.warning(
        """
        **Precision (Alert Quality)**
        - How many flagged transactions were actually fraud?
        - Too many false alarms hurt customers
        """
    )

st.markdown(
    """
    Fraud detection is always a **trade-off**:
    
    🔼 Higher recall → more fraud caught, more false alarms  
    🔼 Higher precision → fewer false alarms, more missed fraud
    """
)

# ===========================
# SECTION 3: WHY ROC-AUC FAILS
# =========================

# ===========================
# SECTION 4: WHY PR-AUC WORKS
# ===========================
st.header("✅ Why PR-AUC Is the Right Metric")

st.markdown(
    """
    **PR-AUC focuses ONLY on the positive class (Fraud)**:
    
    - Precision → Are alerts meaningful?
    - Recall → Are frauds being caught?
    
    👉 No dilution from millions of legitimate transactions
    """
)

st.markdown(
    "ROC-AUC hides the real cost of false positives in imbalanced datasets."

    "PR-AUC directly measures how well the model balances fraud capture and false alarms."
)
st.header("📊 Visual Intuition: ROC vs PR")

np.random.seed(42)

y_true = np.array([0]*995 + [1]*5)
y_scores = np.random.rand(1000)

precision, recall, _ = precision_recall_curve(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

col1, col2 = st.columns(2)

with col1:
    fig = px.line(
        x=fpr,
        y=tpr,
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        title=f"ROC Curve (AUC = {roc_auc:.3f})"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ROC-AUC looks deceptively good.")

with col2:
    fig = px.line(
        x=recall,
        y=precision,
        labels={"x": "Recall", "y": "Precision"},
        title=f"Precision–Recall Curve (AUC = {pr_auc:.3f})"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PR-AUC exposes poor fraud detection.")