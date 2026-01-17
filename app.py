import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Intelligent Customer Segmentation System",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Load Model Pipeline
# -------------------------------
pipeline = joblib.load("intelligent_customer_segmentation.pkl")

scaler = pipeline["scaler"]
pca = pipeline["pca"]
kmeans = pipeline["kmeans"]

# -------------------------------
# App Title & Description
# -------------------------------
st.title("🧠 Intelligent Customer Segmentation & Behavior Analysis System")

st.markdown(
    "This AI-powered system groups customers into meaningful segments using advanced "
    "clustering techniques to support smarter marketing and business decisions."
)

st.divider()

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("📋 Enter Customer Details")

age = st.slider("Customer Age", 18, 80, 30)
income = st.number_input("Annual Income (₹)", 20000, 200000, 60000)
spending = st.slider("Spending Score (1–100)", 1, 100, 50)
visits = st.slider("Monthly Store Visits", 0, 20, 5)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Analyze Customer Segment"):

    # Create input array
    input_data = np.array([[age, income, spending, visits]])

    # Apply preprocessing
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)

    # Predict cluster
    cluster = kmeans.predict(scaled_data)[0]

    # -------------------------------
    # SAFE Dynamic Cluster Mapping
    # -------------------------------
    cluster_labels = {
        0: "Premium Loyal Customers",
        1: "High Income Low Spenders",
        2: "Young Frequent Buyers",
        3: "Budget-Conscious Customers",
        4: "Occasional Low-Engagement Customers"
    }

    # This line prevents future crashes
    segment_name = cluster_labels.get(
        int(cluster),
        f"Customer Segment {int(cluster)}"
    )

    st.success(f"Predicted Customer Segment: **{segment_name}**")

    # -------------------------------
    # Customer Summary Table
    # -------------------------------
    st.markdown("### 🧾 Customer Profile Summary")

    summary = pd.DataFrame({
        "Feature": ["Age", "Annual Income", "Spending Score", "Monthly Visits"],
        "Value": [age, income, spending, visits]
    })

    st.table(summary)

    # -------------------------------
    # PCA Visualization
    # -------------------------------
    st.markdown("### 📊 Customer Position in PCA Space")

    fig, ax = plt.subplots()
    ax.scatter(pca_data[0, 0], pca_data[0, 1], s=120)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("New Customer Position in PCA Space")

    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("Developed by Kali Dharani | Final Year Capstone Project")
