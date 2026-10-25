# Continuous Indian Sign Language Translation Using Deep Learning

An AI-based project that aims to translate continuous Indian Sign Language (ISL) video sequences into meaningful English text using deep learning and multimodal feature extraction.

The backend provides a REST API for video inference, a curated demonstration gallery, video streaming, and model performance information.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Backend](#running-the-backend)
- [API Endpoints](#api-endpoints)
- [Demo Gallery](#demo-gallery)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Dataset and Model Assets](#dataset-and-model-assets)
- [Current Limitations](#current-limitations)

## Project Overview

Communication between Deaf and hearing communities can be challenging when a common language is unavailable.

This project explores the use of deep learning to translate continuous Indian Sign Language videos into English text.

Unlike isolated sign recognition systems that classify individual gestures, continuous sign language translation aims to understand sequences of signs while accounting for temporal context.

The project uses a multimodal approach that combines visual information from video frames with spatial information from hand and body landmarks.

## Objectives

The primary objectives are to:

- Accept ISL video sequences as input.
- Extract visual features using a pretrained VideoMAE model.
- Extract hand and body landmarks using MediaPipe.
- Combine visual and landmark features.
- Model temporal relationships using a neural network architecture.
- Generate English text from the learned representations.
- Provide a REST API for inference and demonstration.
- Support future integration with a web frontend and, eventually, real-time webcam input.

## System Architecture

The current inference pipeline follows this architecture:

```text
              ISL Video
                  |
                  v
          Video Frame Sampling
                  |
          +-------+--------+
          |                |
          v                v
   VideoMAE Features   MediaPipe Landmarks
          |                |
          |          Hand and Body Data
          |                |
          +-------+--------+
                  |
                  v
          Multimodal Fusion
                  |
                  v
          Temporal Encoder
                  |
                  v
        BiGRU-Attention Decoder
                  |
                  v
          Generated English Text
```

### Main Components

**1. Video Processing**

OpenCV reads the input video and samples 32 frames at evenly spaced intervals.

**2. Visual Feature Extraction**

A pretrained VideoMAE model extracts visual representations from the sampled frames. The resulting features are pooled into a sequence of 16 feature vectors.

**3. Landmark Extraction**

MediaPipe extracts pose and hand landmark information across 16 frames:

- Pose: 33 landmarks
- Left hand: 21 landmarks
- Right hand: 21 landmarks

The combined representation contains 75 landmarks per frame, with three coordinates per landmark, giving 225 landmark values per frame.

**4. Multimodal Fusion**

Visual features and landmark information are combined to represent the signing sequence.

**5. Sequence Modeling and Decoding**

The multimodal model uses a temporal encoder and a BiGRU-attention decoder to generate English text token by token.

**6. Backend API**

FastAPI exposes endpoints for health checks, video inference, curated demonstrations, video streaming, and performance information.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Visual feature extraction | VideoMAE |
| Landmark extraction | MediaPipe |
| Video processing | OpenCV |
| Image processing | Pillow |
| Model hosting and downloads | Hugging Face Transformers |
| Backend framework | FastAPI |
| API server | Uvicorn |
| Request validation | Pydantic |
| Testing | Pytest |
| Version control | Git and GitHub |
| Large model storage | Git LFS |

## Repository Structure

The main backend and demonstration files are organized as follows:

```text
continuous-isl-translation/
│
├── data/
│   ├── metadata/
│   │   └── demonstration_gallery.json
│   │
│   ├── processed/
│   │   └── multimodal_run_v5_16steps/
│   │       ├── checkpoints/
│   │       │   └── best.pt
│   │       ├── config.json
│   │       └── vocab.json
│   │
│   └── raw/
│       └── videos/
│           └── Selected demonstration videos
│
├── src/
│   ├── api.py
│   ├── inference.py
│   ├── demo_service.py
│   ├── landmark_extractor.py
│   ├── multimodal_model.py
│   ├── baseline_model.py
│   └── text.py
│
├── tests/
│   ├── test_backend_api.py
│   ├── test_demo_gallery.py
│   └── test_landmark_extractor.py
│
├── .gitattributes
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

Additional training scripts, experiments, and diagnostic files may exist locally but are not all included in the backend commit.

## Getting Started

### Prerequisites

Before running the project, install or prepare:

- Git
- Git LFS
- Python and pip
- The required Python packages
- Internet access for downloading pretrained models if they are not cached locally
- Sufficient RAM and disk space for model loading
- A compatible GPU if GPU acceleration is desired

The model can use CUDA when available and otherwise falls back to CPU execution.

### 1. Clone the Repository

Install Git LFS if it is not already installed.

Then run:

```powershell
git lfs install
git clone https://github.com/Bhavya-Tanna/Continous-ISL-Translation.git
cd Continous-ISL-Translation
```

Git LFS is important because the trained model checkpoint is stored as a large file.

Verify that the checkpoint has been downloaded correctly:

```powershell
git lfs ls-files
```

If the checkpoint appears to be missing or contains an LFS pointer instead of model data, run:

```powershell
git lfs pull
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution policy for your environment or activate the environment through your IDE.

### 3. Install Dependencies

Install the project's Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation requirements can vary depending on the operating system and CUDA version. If GPU acceleration is required, verify that the installed PyTorch build supports your CUDA environment.

### 4. Verify Required Model Assets

The inference pipeline expects the following files to exist:

```text
data/processed/multimodal_run_v5_16steps/
├── checkpoints/
│   └── best.pt
├── config.json
└── vocab.json
```

The pipeline also loads:

- The pretrained VideoMAE model and its image processor.
- The MediaPipe pose model asset.
- The MediaPipe hand model asset.

The pretrained VideoMAE model may be downloaded from Hugging Face if it is not available in the configured local cache.

**Important:** The landmark model asset files must also be available at the paths expected by `src/landmark_extractor.py`. Confirm their locations before launching the service on a new machine.

## Running the Backend

From the repository root, with the virtual environment activated, start the API server:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

For development with automatic code reloading:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

The API should be accessible at:

```text
http://127.0.0.1:8000
```

### Interactive API Documentation

FastAPI generates interactive API documentation automatically.

Open the following address in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the Swagger UI to inspect endpoints, upload videos, and test requests.

### Verify Backend Health

In a second PowerShell terminal, run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

A successful model initialization should return a response with fields such as:

```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "ISLMultimodalModel",
  "device": "cuda",
  "vocab_size": 4024
}
```

This is an illustrative response; the actual checkpoint, device, and vocabulary metadata depend on the running environment.

If initialization fails, inspect the backend terminal logs. The server may start even when the inference model fails to initialize, in which case the health endpoint reports an unhealthy status.

## API Endpoints

The backend exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check backend and model readiness |
| POST | `/predict` | Upload a video for translation |
| GET | `/demo/gallery` | Retrieve curated demo entries |
| GET | `/demo/video/{demo_id}` | Stream a demo video |
| POST | `/demo/predict/{demo_id}` | Retrieve demo information and optionally run inference |
| GET | `/performance` | Retrieve the evaluation summary |

### 1. Health Check

**Endpoint**

```http
GET /health
```

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The response reports model readiness, device information, checkpoint metadata, vocabulary size, supported video extensions, and the maximum upload size.

### 2. Translate an Uploaded Video

**Endpoint**

```http
POST /predict
```

The request uses `multipart/form-data`.

Required field:

- `file`: The ISL video to process.

Optional fields:

- `max_len`: Maximum number of generated tokens; defaults to 30.
- `block_ngram_repeat`: Repetition-blocking window; defaults to 3.

Example using PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@C:\path\to\isl_video.mp4" `
  -F "max_len=30" `
  -F "block_ngram_repeat=3"
```

Replace `C:\path\to\isl_video.mp4` with the actual path to your video.

The response contains:

- Generated translation
- Generated token IDs
- Total processing duration
- VideoMAE, landmark extraction, and decoding timings
- Feature dimensions
- Model metadata
- Decoding safeguard settings

The endpoint accepts MP4, AVI, MOV, MKV, and WebM file extensions, subject to successful video validation.

### 3. Retrieve the Demo Gallery

**Endpoint**

```http
GET /demo/gallery
```

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/demo/gallery
```

The response contains the curated demonstration entries, including demo IDs, video filenames, reference translations, curated translations, categories, and video streaming URLs.

Use the returned `demo_id` values for the following endpoints.

### 4. Stream a Demonstration Video

**Endpoint**

```http
GET /demo/video/{demo_id}
```

Example:

```text
http://127.0.0.1:8000/demo/video/YOUR_DEMO_ID
```

Replace `YOUR_DEMO_ID` with a valid ID from `/demo/gallery`.

This endpoint streams the selected demonstration video.

### 5. Retrieve a Demo Prediction

**Endpoint**

```http
POST /demo/predict/{demo_id}
```

By default, `run_inference` is `false`. The endpoint returns the curated demonstration information without running the model.

Example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/demo/predict/YOUR_DEMO_ID"
```

To request actual model inference:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/demo/predict/YOUR_DEMO_ID?run_inference=true"
```

Replace `YOUR_DEMO_ID` with a valid ID.

The response distinguishes the dataset reference translation, curated demonstration translation, and model prediction when inference is requested and succeeds.

**Note:** A curated demonstration translation is not necessarily the model's prediction. These values are represented separately in the API response.

### 6. Retrieve the Performance Summary

**Endpoint**

```http
GET /performance
```

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/performance
```

This endpoint returns the performance summary available through the demonstration service.

Refer to the returned metrics and their documented definitions when evaluating model quality.

## Demo Gallery

The repository includes eight selected demonstration videos and a gallery metadata file:

```text
data/
├── metadata/
│   └── demonstration_gallery.json
└── raw/
    └── videos/
        └── Selected MP4 files
```

The gallery allows a frontend or evaluator to discover demonstration videos and compare available reference, curated, and model-generated text.

The demo endpoints can be used independently of a frontend, making it possible to test the backend through Swagger UI or HTTP requests.

## Testing

The backend includes tests for API behavior, the demonstration gallery, and landmark extraction.

Run the focused test suite from the repository root:

```powershell
python -m pytest tests/test_backend_api.py tests/test_demo_gallery.py tests/test_landmark_extractor.py -q
```

To run the entire available test suite:

```powershell
python -m pytest -q
```

Some tests may depend on installed packages, model assets, or environment-specific configuration.

A successful test run verifies the behavior covered by those tests; it does not, by itself, establish translation accuracy or guarantee successful inference on every machine.

## Configuration

The backend supports environment variables for inference settings.

| Variable | Default | Purpose |
|---|---|---|
| `ISL_MAX_UPLOAD_MB` | `50` | Maximum uploaded video size in MB |
| `ISL_DEFAULT_MAX_LEN` | `30` | Default maximum generated token count |
| `ISL_BLOCK_NGRAM` | `3` | Default n-gram repetition-blocking window |
| `ISL_CHECKPOINT_PATH` | Default checkpoint path | Override the trained checkpoint location |
| `ISL_CONFIG_PATH` | Default config path | Override the model configuration path |
| `ISL_VOCAB_PATH` | Default vocabulary path | Override the vocabulary file |
| `ISL_DEVICE` | Automatically selected | Explicitly select the inference device |

Example PowerShell configuration:

```powershell
$env:ISL_DEVICE = "cpu"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Use `cuda` instead of `cpu` when the installed PyTorch environment supports CUDA and a compatible GPU is available.

These environment variables configure the existing backend; they do not automatically download missing model assets or create a compatible model checkpoint.

## Troubleshooting

### Model fails to initialize

Check that the following files exist:

```text
data/processed/multimodal_run_v5_16steps/checkpoints/best.pt
data/processed/multimodal_run_v5_16steps/config.json
data/processed/multimodal_run_v5_16steps/vocab.json
```

If the checkpoint was downloaded through Git LFS, verify that it contains the actual model data.

Also confirm that the pretrained VideoMAE and MediaPipe assets are available.

### Hugging Face download fails

If VideoMAE is not cached locally, the pipeline may attempt to download it from Hugging Face.

Check your internet connection, available disk space, and access to the model repository. Retry after resolving any download or connectivity issues.

### API reports an unhealthy status

Request:

```text
http://127.0.0.1:8000/health
```

Review the terminal logs for checkpoint-loading errors, missing configuration files, missing vocabulary files, or failures involving pretrained models and landmark assets.

### Video upload fails

Verify that:

- The file uses a supported video extension.
- The video is not empty or corrupted.
- OpenCV can read its frames.
- The file does not exceed the configured upload-size limit.

### CUDA-related errors

Check your installed PyTorch build and GPU compatibility. If GPU execution is unavailable, try running with:

```powershell
$env:ISL_DEVICE = "cpu"
```

CPU inference may be significantly slower.

## Dataset and Model Assets

The project uses the iSign Indian Sign Language dataset for experimentation.

Dataset preparation has included metadata inspection, filtering, subset selection, video retrieval, integrity validation, and dataset characterization.

The repository contains selected demonstration videos and the model assets required by the documented inference pipeline. It does not contain the entire original dataset.

The pretrained VideoMAE model and required MediaPipe assets may need to be downloaded or configured separately.

Use the original dataset and pretrained models according to their respective licenses and usage terms.

## Current Limitations

This project is under active development.

Important considerations include:

- Translation quality depends on the trained model and the input video.
- The model's output should not be assumed to be a correct translation without evaluation.
- Curated demonstration text must be distinguished from actual model predictions.
- Inference may require substantial memory and computational resources.
- Pretrained model and MediaPipe asset availability affects deployment.
- Real-time webcam integration remains a future development objective.

## Future Improvements

Potential areas of improvement include:

- Improving translation quality and generalization.
- Expanding evaluation across held-out videos.
- Optimizing inference latency and memory consumption.
- Integrating the backend with a web frontend.
- Developing real-time webcam inference.
- Exploring improved multimodal fusion and temporal modeling techniques.

## License and Responsible Dataset Usage

The dataset remains subject to its original license and usage restrictions.

Do not redistribute the original dataset or downloaded dataset files without appropriate authorization.

Refer to the original dataset and pretrained model documentation for their respective terms of use.