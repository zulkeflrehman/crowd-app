"""
Crowd Counting App — powered by a Roboflow-hosted object detection model.
Model: crowd-density-ou3ne (Roboflow Universe, by abraham)
Confidence threshold is FIXED at 30% (0.30) — not user-adjustable.
Detection cap raised to 1000 (default is 300).
"""

import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# --------------------------------------------------------------------------
MODEL_ID = "crowd-density-ou3ne/1"      # <-- double-check the version number
CONFIDENCE_THRESHOLD = 0.20             # fixed at 30%
MAX_DETECTIONS = 1500                   # raised above the default 300 cap
API_URL = "https://serverless.roboflow.com"

st.set_page_config(page_title="Crowd Counting (Roboflow)", layout="wide")


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
    preds = result.get("predictions", [])
    return preds, len(preds)


def draw_predictions(pil_image: Image.Image, preds: list) -> Image.Image:
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for p in preds:
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
        draw.text((x0, max(0, y0 - 12)), f"{p.get('confidence', 0):.2f}", fill="lime")
    return img


st.title("Crowd Counting")
st.caption(
    f"Model: `{MODEL_ID}`  •  Confidence fixed at {int(CONFIDENCE_THRESHOLD * 100)}%  "
    f"•  Detection cap: {MAX_DETECTIONS}"
)

client = get_client()

tab_img, tab_video = st.tabs(["Image", "Video"])

with tab_img:
    uploaded_img = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"], key="img_uploader")
    if uploaded_img is not None:
        pil_img = Image.open(uploaded_img).convert("RGB")
        with st.spinner("Running detection..."):
            preds, count = run_inference(client, pil_img)
            annotated = draw_predictions(pil_img, preds)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original", use_container_width=True)
        with col2:
            st.image(annotated, caption="Detections", use_container_width=True)

        st.metric("Estimated Crowd Count", count)

with tab_video:
    st.info("Video mode calls the API once per sampled frame — kept small by default so it stays fast.")

    uploaded_video = st.file_uploader("Upload a crowd video", type=["mp4", "avi", "mov"], key="video_uploader")
    sample_every_n_seconds = st.slider("Sample one frame every N seconds", 1, 10, 1)
    max_frames = st.slider("Max frames to process", 5, 100, 30)

    if uploaded_video is not None and st.button("Run video analysis"):
        import cv2
        import tempfile

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, int(round(fps * sample_every_n_seconds)))

        counts, timestamps = [], []
        frame_idx, processed = 0, 0
        progress = st.progress(0)
        status = st.empty()

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                _, count = run_inference(client, pil_frame)
                counts.append(count)
                timestamps.append(frame_idx / fps)
                processed += 1
                progress.progress(processed / max_frames)
                status.text(f"Frame {frame_idx} (t={frame_idx/fps:.1f}s): count = {count}")
            frame_idx += 1

        cap.release()
        os.unlink(tfile.name)

        if counts:
            avg_count = float(np.mean(counts))
            max_count = int(np.max(counts))
            peak_time = timestamps[int(np.argmax(counts))]

            c1, c2, c3 = st.columns(3)
            c1.metric("Frames Processed", len(counts))
            c2.metric("Average Count", f"{avg_count:.1f}")
            c3.metric("Maximum Count", max_count, help=f"at ~{peak_time:.1f}s")

            st.line_chart({"Estimated Count": counts}, use_container_width=True)
        else:
            st.warning("No frames were processed — check the video file.")
