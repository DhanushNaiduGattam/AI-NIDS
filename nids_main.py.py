# ============================================================
# AI-Based Network Intrusion Detection System
# Edunet-VOIS Internship Final Project
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
CSV_NAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
FEATURES = ["packet_size", "duration", "src_bytes", "dst_bytes", "flag_count"]

# ------------------------------------------------------------
# CREATE SAMPLE CSV IF MISSING
# ------------------------------------------------------------
def create_sample_csv():
    if not os.path.exists(CSV_NAME):
        np.random.seed(42)
        size = 1000
        df = pd.DataFrame({
            "packet_size": np.random.randint(40, 1500, size),
            "duration": np.random.uniform(0.01, 5.0, size),
            "src_bytes": np.random.randint(0, 50000, size),
            "dst_bytes": np.random.randint(0, 50000, size),
            "flag_count": np.random.randint(0, 10, size),
            "label": np.random.choice([0, 1], size)
        })
        df.to_csv(CSV_NAME, index=False)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
def load_data(mode):
    if mode == "Simulation":
        np.random.seed(42)
        size = 2000
        return pd.DataFrame({
            "packet_size": np.random.randint(40, 1500, size),
            "duration": np.random.uniform(0.01, 5.0, size),
            "src_bytes": np.random.randint(0, 50000, size),
            "dst_bytes": np.random.randint(0, 50000, size),
            "flag_count": np.random.randint(0, 10, size),
            "label": np.random.choice([0, 1], size)
        })

    create_sample_csv()
    return pd.read_csv(CSV_NAME)

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------
def train_model(df):
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    feature_importance = model.feature_importances_
    
    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return model, acc, report, cm, feature_importance, fpr, tpr, roc_auc

# ------------------------------------------------------------
# DETECT TRAFFIC
# ------------------------------------------------------------
def detect(model, values):
    data = pd.DataFrame([values], columns=FEATURES)
    pred = model.predict(data)[0]
    prob = model.predict_proba(data).max() * 100
    return pred, prob

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.set_page_config("AI-Based Network Intrusion Detection System", layout="wide")
st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("**Edunet-VOIS Internship Final Project** - A machine learning-powered system for detecting network intrusions using Random Forest classification.")

st.sidebar.header("🔧 Controls")
mode = st.sidebar.selectbox("Select Data Mode", ["Simulation", "CSV"])

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.df = None
    st.session_state.report = None
    st.session_state.cm = None
    st.session_state.feature_importance = None
    st.session_state.fpr = None
    st.session_state.tpr = None
    st.session_state.roc_auc = None

if st.sidebar.button("🚀 Train Model"):
    df = load_data(mode)
    st.session_state.df = df
    st.session_state.model, acc, report, cm, feature_importance, fpr, tpr, roc_auc = train_model(df)
    st.session_state.report = report
    st.session_state.cm = cm
    st.session_state.feature_importance = feature_importance
    st.session_state.fpr = fpr
    st.session_state.tpr = tpr
    st.session_state.roc_auc = roc_auc
    st.sidebar.success("Model trained successfully")
    st.sidebar.metric("Accuracy", f"{acc*100:.2f}%")

# ------------------------------------------------------------
# INPUT FIELDS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Traffic Analysis", "📈 Data Visualization", "📝 Review"])

with tab1:
    st.header("System Overview")
    st.write("This system uses a Random Forest classifier trained on network traffic features to detect intrusions.")
    st.info("**How it works:** The model analyzes network packet data to classify traffic as normal or intrusive. Train the model first, then input live traffic features for analysis.")
    if st.session_state.report:
        st.subheader("Model Performance Metrics")
        st.write("These metrics evaluate the model's ability to detect intrusions accurately.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision (Intrusion)", f"{st.session_state.report['1']['precision']:.2f}")
            st.write("Precision: Fraction of detected intrusions that are actual intrusions.")
            st.metric("Recall (Intrusion)", f"{st.session_state.report['1']['recall']:.2f}")
            st.write("Recall: Fraction of actual intrusions that are detected.")
        with col2:
            st.metric("F1-Score (Intrusion)", f"{st.session_state.report['1']['f1-score']:.2f}")
            st.write("F1-Score: Harmonic mean of precision and recall.")
            st.metric("AUC-ROC", f"{st.session_state.roc_auc:.2f}")
            st.write("AUC-ROC: Area under the ROC curve, measuring model's discriminative ability.")
        
        st.subheader("Advanced Visualizations")
        st.write("Visual insights into model behavior and data patterns.")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            # Feature Importance
            if st.session_state.feature_importance is not None:
                fig_imp = px.bar(x=FEATURES, y=st.session_state.feature_importance, title="Feature Importance")
                st.plotly_chart(fig_imp)
                st.write("Feature Importance: Shows which traffic features contribute most to predictions.")
        
        with viz_col2:
            # Confusion Matrix
            if st.session_state.cm is not None:
                cm_df = pd.DataFrame(st.session_state.cm, index=["Normal", "Intrusion"], columns=["Predicted Normal", "Predicted Intrusion"])
                fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
                st.plotly_chart(fig_cm)
                st.write("Confusion Matrix: Breakdown of correct and incorrect predictions.")
        
        # ROC Curve
        if st.session_state.fpr is not None and st.session_state.tpr is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=st.session_state.fpr, y=st.session_state.tpr, mode='lines', name=f'ROC Curve (AUC = {st.session_state.roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc)
            st.write("ROC Curve: Plots true positive rate vs. false positive rate at different thresholds.")

with tab2:
    st.header("🔴 Live Traffic Analysis")
    st.write("Input network traffic parameters to analyze for potential intrusions.")
    col1, col2, col3 = st.columns(3)
    with col1:
        packet_size = st.number_input("Packet Size", 40, 1500, 500, help="Size of the network packet in bytes.")
        duration = st.number_input("Flow Duration", 0.01, 10.0, 1.0, help="Duration of the network flow in seconds.")
    with col2:
        src_bytes = st.number_input("Source Bytes", 0, 100000, 2000, help="Bytes sent from source.")
        dst_bytes = st.number_input("Destination Bytes", 0, 100000, 1500, help="Bytes sent to destination.")
    with col3:
        flag_count = st.number_input("Flag Count", 0, 20, 2, help="Number of TCP flags set.")

    if st.button("🔍 Analyze Traffic"):
        if st.session_state.model is None:
            st.warning("⚠ Please train the model first")
        else:
            result, confidence = detect(
                st.session_state.model,
                [packet_size, duration, src_bytes, dst_bytes, flag_count]
            )

            if result == 1:
                st.error(f"🚨 INTRUSION DETECTED ({confidence:.2f}%)")
                st.write("High confidence indicates potential security threat.")
            else:
                st.success(f"✅ NORMAL TRAFFIC ({confidence:.2f}%)")
                st.write("Traffic appears normal based on model prediction.")

with tab3:
    st.header("Data Visualization")
    st.write("Explore the training data distribution and patterns.")
    if st.session_state.df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(st.session_state.df, x="packet_size", color="label", title="Packet Size Distribution")
            st.plotly_chart(fig_hist)
        with col2:
            fig_box = px.box(st.session_state.df, x="label", y="src_bytes", title="Source Bytes by Label")
            st.plotly_chart(fig_box)
        
        fig_scatter = px.scatter(st.session_state.df, x="packet_size", y="src_bytes", color="label", title="Packet Size vs Source Bytes")
        st.plotly_chart(fig_scatter)
        st.write("Simple plots to understand data distribution and relationships.")
    else:
        st.info("Train the model to view data visualization.")

with tab4:
    st.header("📝 Project Review")
    st.write("Provide feedback on the AI-Based Network Intrusion Detection System.")
    st.subheader("Project Summary")
    st.write("""
    - **Objective:** Develop an ML-powered NIDS using Random Forest.
    - **Features:** Model training, live traffic analysis, performance metrics, visualizations.
    - **Dataset:** Simulated or CSV-based network traffic data.
    - **Technologies:** Python, Streamlit, Scikit-learn, Plotly.
    """)
    feedback = st.text_area("Your Feedback", placeholder="Share your thoughts on the project...")
    if st.button("Submit Review"):
        st.success("Thank you for your feedback!")
        st.write(f"Submitted: {feedback}")

st.markdown("---")
st.caption("Dataset: CIC-IDS2017 | Algorithm: Random Forest | Edunet-VOIS Internship Project")