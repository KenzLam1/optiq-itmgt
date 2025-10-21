# Optiq Retail Analytics

Optiq Retail Analytics is a desktop application that detects shoppers in live or recorded video feeds and estimates their age and gender. The app offers three capture modes:
- **Webcam** - use the default camera connected to your machine.
- **Video file** - analyse any pre-recorded MP4/AVI/MKV file.
- **CCTV / RTSP stream** - connect to an IP camera or NVR stream (for example `rtsp://user:pass@host:554/stream`).

## Prerequisites

1. Install Python 3.9 or newer.
2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download weights `person_dector.pt` and `age-gender_detector.pt` and place them beside `main.py`. Weights are hosted in Releases (tag weights-v1).

> **Note:** PyTorch automatically uses the GPU if one is available; otherwise it falls back to CPU execution.

## Running the application

```bash
python main.py
```

Use the control panel on the left to choose a video source and click **Start Analysis**.

## Troubleshooting

- If the age & gender overlays never appear, double-check that `age-gender_detector.pt` matches the expected model format.
- If the app fails to launch complaining about `person_detector.pt`, download a YOLO COCO model (for example `yolov8n.pt`) and rename it as instructed above.
- For RTSP streams, ensure the URL is reachable from your machine and that firewalls allow the traffic.
- On lower powered machines, increase the *Frame Skip* value in the settings drawer to reduce CPU load.

## License

This project is provided as-is for internal evaluation at Optiq Retail.
