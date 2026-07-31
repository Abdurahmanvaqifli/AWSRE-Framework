import streamlit as st

st.set_page_config(
    page_title="AWSRE Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AWSRE Dashboard")
st.caption("Adaptive Watermarking Strategy Recommendation Engine")

st.divider()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Algorithms",
        "3",
        "DCT • DWT • DCT-SVD",
    )

with c2:
    st.metric(
        "Framework",
        "v1.0",
    )

with c3:
    st.metric(
        "Pages",
        "6",
    )

with c4:
    st.metric(
        "Status",
        "Ready",
    )

st.divider()

st.subheader("Framework Modules")

left, right = st.columns(2)

with left:
    st.success("✅ Invisible Watermark Embedding")
    st.success("✅ Watermark Extraction")
    st.success("✅ Benchmark Evaluation")

with right:
    st.success("✅ Recommendation Engine")
    st.success("✅ Strategy Explorer")
    st.success("✅ Experimental Analysis")

st.divider()

st.subheader("Available Algorithms")

st.markdown("""
| Method | Status |
|--------|--------|
| DCT | ✅ |
| DWT | ✅ |
| DCT-SVD | ✅ |
| DWT-SVD | 🚧 |
| Block-SVD | 🚧 |
""")

st.divider()

st.info(
    "Use the left sidebar to navigate through the AWSRE Framework."
)
