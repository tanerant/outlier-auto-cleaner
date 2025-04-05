import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from io import StringIO

st.set_page_config(page_title="Outlier Auto-Cleaner", page_icon="🧹")
st.title("🧹 Outlier Auto-Cleaner")
st.write("Upload a CSV, choose a method, and clean out those pesky outliers!")

# Helper functions
def remove_outliers_zscore(df, threshold=3.0):
    numeric_df = df.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(numeric_df))
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]

def remove_outliers_iqr(df):
    numeric_df = df.select_dtypes(include=np.number)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask]

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Raw Data Preview", df.head())
    st.write(f"✅ {df.shape[0]} rows, {df.shape[1]} columns")

    method = st.selectbox("Choose outlier detection method:", ["Z-score", "IQR"])

    if method == "Z-score":
        threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0)
        cleaned_df = remove_outliers_zscore(df, threshold)
    else:
        cleaned_df = remove_outliers_iqr(df)

    st.write("### 🧼 Cleaned Data Preview", cleaned_df.head())
    st.write(f"📉 Rows after cleaning: {cleaned_df.shape[0]} (removed {df.shape[0] - cleaned_df.shape[0]})")

    # Download button
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name='cleaned_data.csv',
        mime='text/csv'
    )

    st.success("Done! Cleaned data ready to use.")

else:
    st.info("👆 Upload a CSV file to begin.")
