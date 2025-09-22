# xgboost_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("XGBoost Model for Debt-Based Crowdfunding Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Ask user for target column
    target_col = st.selectbox("Select Target Column (label)", df.columns)

    # Features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------------------------
    # Train-test split
    # ---------------------------
    # Check if stratify is possible (at least 2 classes in target)
    if y.nunique() > 1:
        stratify_param = y
    else:
        stratify_param = None
        st.warning("Target has only 1 class. Stratification disabled for train-test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=stratify_param, random_state=42
    )

    # ---------------------------
    # Train XGBoost Classifier
    # ---------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ---------------------------
    # Metrics Calculation
    # ---------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if y.nunique() > 1 else 0.0

    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame({
        "Accuracy": [acc * 100],
        "Precision": [prec * 100],
        "Recall": [rec * 100],
        "F1-score": [f1 * 100],
        "ROC-AUC": [auc]
    })
    st.dataframe(metrics_df.style.format("{:.2f}"))

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    st.pyplot(fig_cm)

    # ---------------------------
    # ROC Curve
    # ---------------------------
    if y.nunique() > 1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # ---------------------------
    # Feature Importance
    # ---------------------------
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    sns.barplot(data=feat_df.head(15), x="Importance", y="Feature", ax=ax_imp, palette="viridis")
    ax_imp.set_title("Top 15 Features")
    st.pyplot(fig_imp)

else:
    st.info("👆 Upload a CSV file to begin.")

