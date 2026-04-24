import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Recycling Classifier", page_icon="♻️", layout="centered")

st.title("♻️ Recycling Classifier")
st.write("Upload an image. The app sends it to your FastAPI backend and displays the prediction. v. 12.4")

API_URL = os.getenv("BACKEND_PREDICT_URL", "http://localhost:8000/predict")
TIMEOUT_SECONDS = 60


@st.cache_data(show_spinner=False)
def load_image_preview(file_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(file_bytes))


uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

st.caption(f"Current backend: `{API_URL}`")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    preview = load_image_preview(file_bytes)
    st.image(preview, caption="Preview", use_container_width=True)

    if st.button("Analyze image", type="primary"):
        with st.spinner("Sending image to backend and waiting for response..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type or "application/octet-stream",
                    )
                }

                response = requests.post(API_URL, files=files, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                result = response.json()

                st.success("Analysis completed successfully")

                material = result.get("material", "Unknown")
                confidence = result.get("confidence")
                more_info = result.get("more_info_url")
                recycle_info = result.get("recycle_url")

                st.subheader("Result")
                st.write(f"**Material:** {material}")

                if confidence is not None:
                    try:
                        st.write(f"**Confidence:** {float(confidence):.2%}")
                    except (TypeError, ValueError):
                        st.write(f"**Confidence:** {confidence}")

                if more_info:
                    st.markdown(f"**More information:** [Link]({more_info})")

                if recycle_info:
                    st.markdown(f"**Where to recycle:** [Link]({recycle_info})")

                with st.expander("Raw API response"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend. Check whether FastAPI is running and the URL is correct."
                )
            except requests.exceptions.Timeout:
                st.error("The request to the backend timed out.")
            except requests.exceptions.HTTPError as exc:
                error_text = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Backend error: {error_text}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
else:
    st.info("Please upload an image to start the analysis.")