"""
AWSRE Framework
Main Streamlit Application
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AWSRE Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ AWSRE Framework")
st.subheader("Adaptive Watermarking Strategy Recommendation Engine")

st.markdown("---")

st.markdown(
"""
Welcome to the AWSRE Framework.

This platform provides:

- 🔹 Invisible image watermark embedding
- 🔹 Watermark extraction
- 🔹 Benchmark comparison
- 🔹 Intelligent recommendation engine
- 🔹 Experimental evaluation
"""
)

st.info(
    "Select a page from the left sidebar to begin."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Implemented Methods",
        "3",
        "DCT • DWT • DCT-SVD",
    )

with col2:
    st.metric(
        "Framework Version",
        "1.0",
    )

with col3:
    st.metric(
        "Status",
        "Development",
    )

st.markdown("---")

st.success("AWSRE Framework loaded successfully.")
