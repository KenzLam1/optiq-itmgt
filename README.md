# Optiq Retail Analytics

Optiq Retail Analytics detects shoppers in live or recorded video feeds and estimates age and gender. Detections are then turned into useful foot traffic data through various graphs in the dashboard. Filters allow the user to track detection data across a specified time and demographics. In short, the project brings to life the idea of having something like Google Analytics for physical establishments.

Currently an unfinished MVP.

## Prerequisites

1. Install Python 3.9 or newer.
2. Install the project dependencies:
  
   a. pip install -r requirements.txt
   
   b. Install pytorch CUDA enabled or MPS supported depending on your OS
   
   For mac:
   ```bash
   pip install torch torchvision torchaudio
   ```
   For windows (it depends on GPU):
   https://www.youtube.com/watch?v=M60_J-jjtn0
      
3. Place the model weights (`age-gender_detector.pt` and `person_detector.pt`) in the project directory. Our trained models can be downloaded from releases (see Initial model weights, tag: weights-v1).

## Running the dashboard

In the project directory, run in the terminal:

```bash
streamlit run main.py
```

Your browser will open to `http://localhost:8501`. Use the sidebar to select the capture source and model settings, then click **Run analysis**.

### Capture modes

- **Upload video** – upload an MP4/MOV/AVI/MKV clip directly from the browser.
- **Webcam** – process frames from a locally connected camera.

### Output

After each run the dashboard surfaces:

- The latest annotated frame with bounding boxes and age/gender overlays.
- Frame, detection count, and throughput metrics.
- A sortable detections table with confidence, age estimates, and bounding boxes.
- A CSV export button for downstream analysis.

## Troubleshooting

- If the page reports missing weights, verify both `.pt` files exist next to `main.py` or provide absolute paths in the sidebar.
- If pytorch isnt detecting CUDA you might want to:

```bash
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

- If pytorch isnt detecting MPS you might want to:

```bash
pip3 install torch torchvision
```

- For Webcam sources, ensure the Streamlit process has permission to access the device or network.
- Use the **Run age/gender detector** and **Run person detector** toggles to keep only the model(s) you need; disabling one frees compute, but keep at least one enabled before running analysis.

## License

dont steal bozo
