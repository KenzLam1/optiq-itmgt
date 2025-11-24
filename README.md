# Optiq Retail Analytics

Optiq Retail Analytics now ships as a Streamlit dashboard that detects shoppers in live or recorded video feeds and estimates age and gender. The web UI reuses the YOLO-based pipelines from the desktop build and adds rich summaries, annotated previews, and CSV exports.

## Prerequisites

1. Install Python 3.9 or newer.
2. Install the project dependencies:
  
   a. pip install -r requirements.txt
   b. Install pytorch CUDA enabled or MPS supported depending on your OS
 
3. Place the model weights (`age-gender_detector.pt` and `person_detector.pt`) alongside `main.py`.

## Running the dashboard

Start Streamlit in the project directory:

```bash
streamlit run main.py
```

Your browser will open to `http://localhost:8501`. Use the sidebar to select the capture source and model settings, then click **Run analysis**.

### Capture modes

- **Upload video** – upload an MP4/MOV/AVI/MKV clip directly from the browser.
- **Webcam** – process frames from a locally connected camera.

The `Max frames to analyse` control limits how many frames are processed per run, which keeps long-running streams manageable.

### Output

After each run the dashboard surfaces:

- The latest annotated frame with bounding boxes and age/gender overlays.
- Frame, detection count, and throughput metrics.
- A sortable detections table with confidence, age estimates, and bounding boxes.
- A CSV export button for downstream analysis.

## Troubleshooting

- If the page reports missing weights, verify both `.pt` files exist next to `main.py` or provide absolute paths in the sidebar.
- For Webcam sources, ensure the Streamlit process has permission to access the device or network.
- Use the **Run age/gender detector** and **Run person detector** toggles to keep only the model(s) you need; disabling one frees compute, but keep at least one enabled before running analysis.
- Uploaded videos are written to a temporary file during the session and deleted once processing completes.

## License

This project is provided as-is for internal evaluation at Optiq Retail.
