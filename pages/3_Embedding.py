import io

import streamlit as st
import numpy as np
from PIL import Image

from watermarking.registry import create_watermarker


st.set_page_config(
    page_title="Embedding",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Invisible Watermark Embedding")

st.markdown("---")

left, right = st.columns(2)

with left:

    host_file = st.file_uploader(
        "Upload Host Image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

with right:

    watermark_file = st.file_uploader(
        "Upload Watermark",
        type=["png", "jpg", "jpeg", "bmp"],
    )

st.markdown("---")

method = st.selectbox(

    "Watermarking Method",

    [

        "DCT",

        "DWT",

        "DCT-SVD",

    ],

)

alpha = st.slider(

    "Embedding Strength (Alpha)",

    1,

    50,

    20,

)

if host_file is not None:

    host_image = Image.open(host_file).convert("L")

    st.image(

        host_image,

        caption="Host Image",

        use_container_width=True,

    )

if watermark_file is not None:

    watermark_image = Image.open(

        watermark_file

    ).convert("L")

    st.image(

        watermark_image,

        caption="Watermark",

        width=200,

    )

st.markdown("---")

if st.button("🚀 Embed Watermark"):

    if host_file is None:

        st.error("Please upload a host image.")

        st.stop()

    if watermark_file is None:

        st.error("Please upload a watermark.")

        st.stop()

    host = np.array(host_image)

    watermark = np.array(watermark_image)

    try:

        algorithm = create_watermarker(

            method,

            alpha=alpha,

        )

        result = algorithm.embed(

            host,

            watermark,

        )

        watermarked = result.watermarked_image

        st.success("Embedding completed successfully.")

        st.image(

            watermarked,

            caption="Watermarked Image",

            use_container_width=True,

        )

        output = Image.fromarray(

            watermarked

        )

        buffer = io.BytesIO()

        output.save(

            buffer,

            format="PNG",

        )

        st.download_button(

            "⬇ Download Watermarked Image",

            data=buffer.getvalue(),

            file_name="watermarked.png",

            mime="image/png",

        )

        with st.expander("Metadata"):

            st.json(result.metadata)

    except Exception as error:

        st.exception(error)
