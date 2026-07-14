# Crowd Counting — Roboflow + Streamlit

Uses the Roboflow-hosted `crowd-density-ou3ne` object detection model
(no training needed). Confidence fixed at 30%, detection cap raised to 1000.

## Run locally
```
pip install -r requirements.txt
# put your key in .streamlit/secrets.toml first
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a **public** GitHub repo (never commit `secrets.toml`).
2. Go to https://share.streamlit.io -> New app -> pick the repo/branch -> `app.py`.
3. In **Advanced settings -> Secrets**, paste:
   ```
   ROBOFLOW_API_KEY = "your_real_key"
   ```
4. Click **Deploy**.
