"""
Crowd Gender Counting App — powered by a Roboflow-hosted object detection model.
Model: gender-detection-ew3rr (Roboflow Universe, by "c")
Confidence fixed at 30%, detection cap raised to 1000. Image only — no video mode.
"""

import os
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

MODEL_ID = "gender-detection-ew3rr/1"
CONFIDENCE_THRESHOLD = 0.30
MAX_DETECTIONS = 1000
API_URL = "https://serverless.roboflow.com"
BOX_COLORS = {"Male": "#3498db", "Female": "#e84393"}

st.set_page_config(page_title="Crowd Gender Counting", layout="wide")


@st.cache_resource
def get_client():
    api_key = st.secrets.get("ROBOFLOW_API_KEY", os.environ.get("ROBOFLOW_API_KEY", ""))
    if not api_key:
        st.error(
            "No Roboflow API key found. Add it to `.streamlit/secrets.toml` locally, "
            "or in the app's 'Secrets' box on Streamlit Community Cloud."
        )
        st.stop()

    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
    client.configure(InferenceConfiguration(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_detections=MAX_DETECTIONS,
    ))
    return client


def run_inference(client, pil_image: Image.Image):
    result = client.infer(pil_image, model_id=MODEL_ID)
    return result.get("predictions", [])


def draw_predictions(pil_image: Image.Image, preds: list) -> Image.Image:
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for p in preds:
        cls = p.get("class", "Unknown")
        color = BOX_COLORS.get(cls, "yellow")
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(0, y0 - 12)), f"{cls} {p.get('confidence', 0):.2f}", fill=color)
    return img


st.title("Crowd Gender Counting")
st.caption(
    f"Model: `{MODEL_ID}`  •  Confidence fixed at {int(CONFIDENCE_THRESHOLD * 100)}%  "
    f"•  Detection cap: {MAX_DETECTIONS}"
)

client = get_client()

uploaded_img = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    pil_img = Image.open(uploaded_img).convert("RGB")

    with st.spinner("Running detection..."):
        preds = run_inference(client, pil_img)
        annotated = draw_predictions(pil_img, preds)

    counts = Counter(p.get("class", "Unknown") for p in preds)
    male_count = counts.get("Male", 0)
    female_count = counts.get("Female", 0)
    total_count = len(preds)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Detections", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Male", male_count)
    m2.metric("Female", female_count)
    m3.metric("Total", total_count)
