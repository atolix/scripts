# Face Tracking

A local real-time face tracking script for TouchDesigner-related workflows.

This project uses MediaPipe Face Landmarker and OpenCV to detect facial landmarks from a webcam feed.

## Project Structure

```text
face-tracking/
├── face_tracking.py
├── requirements.txt
├── README.md
├── models/
│   └── face_landmarker.task
└── .venv/
```

The `.venv` directory and model files should not be committed to Git.

## Requirements

* Python 3.9 or later
* A webcam
* macOS, Windows, or Linux

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download the Face Landmarker Model

Create the model directory:

```bash
mkdir -p models
```

Download the MediaPipe Face Landmarker model:

```bash
curl -L \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o models/face_landmarker.task
```

The final path should be:

```text
models/face_landmarker.task
```

## Run

```bash
python face_tracking.py
```

A webcam window should open and display facial landmarks over detected faces.

Press `Q` to quit.

## macOS Camera Permission

On macOS, the terminal application may request permission to access the camera.

If the permission prompt does not appear, reset the camera permission:

```bash
tccutil reset Camera
```

Then completely quit and reopen the terminal application before running the script again.

## Configuration

The following values can be changed in `face_tracking.py`:

```python
CAMERA_INDEX = 0
MAX_FACES = 1
```

Change `CAMERA_INDEX` to `1` or `2` when using another camera device.

## Planned TouchDesigner Integration

Possible future additions include:

* Sending face coordinates to TouchDesigner over OSC
* Generating a face mask
* Tracking facial expressions
* Tracking multiple faces
* Applying effects only inside the detected face region

