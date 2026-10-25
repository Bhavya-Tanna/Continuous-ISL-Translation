"""FastAPI Backend Service for Continuous Indian Sign Language Translation.

Endpoints:
- GET /health: Health check and model readiness probe.
- POST /predict: Upload an ISL video file and receive the translated English text,
                 processing latency, feature dimensions, and execution breakdown.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.demo_service import DemoGalleryService
from src.inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_VOCAB,
    ISLInferencePipeline,
)

logger = logging.getLogger("isl.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Configuration from Environment
MAX_UPLOAD_SIZE_MB = int(os.environ.get("ISL_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_MAX_LEN = int(os.environ.get("ISL_DEFAULT_MAX_LEN", "30"))
DEFAULT_BLOCK_NGRAM = int(os.environ.get("ISL_BLOCK_NGRAM", "3"))

# Pipeline container
pipeline: Optional[ISLInferencePipeline] = None
demo_service = DemoGalleryService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load heavy model pipeline once at startup."""
    global pipeline
    logger.info("Initializing ISL backend pipeline at startup...")
    demo_service.reload_gallery()
    ckpt_path = os.environ.get("ISL_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT))
    cfg_path = os.environ.get("ISL_CONFIG_PATH", str(DEFAULT_CONFIG))
    voc_path = os.environ.get("ISL_VOCAB_PATH", str(DEFAULT_VOCAB))
    dev = os.environ.get("ISL_DEVICE", None)

    try:
        pipeline = ISLInferencePipeline(
            checkpoint_path=ckpt_path,
            config_path=cfg_path,
            vocab_path=voc_path,
            device=dev,
        )
        logger.info("ISL backend pipeline successfully loaded and ready.")
    except Exception as e:
        logger.error("Failed to initialize pipeline at startup: %s", e, exc_info=True)
        pipeline = None

    yield

    logger.info("Shutting down ISL backend pipeline...")
    if pipeline is not None:
        pipeline.close()
        pipeline = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Continuous Indian Sign Language Translation API",
    description=(
        "Backend API serving continuous Indian Sign Language translation "
        "using Multimodal VideoMAE (16 steps) + MediaPipe Landmarks (75 points) "
        "with BiGRU-Attention decoding."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web frontend consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ready' or 'unhealthy'")
    model_loaded: bool
    model_name: str
    checkpoint: str
    checkpoint_epoch: Optional[int]
    checkpoint_val_loss: Optional[float]
    device: str
    vocab_size: int
    max_upload_size_mb: int
    allowed_extensions: list[str]


class StagesMs(BaseModel):
    videomae_ms: float
    landmarks_ms: float
    decode_ms: float


class FeatureShapes(BaseModel):
    rgb_features: list[int]
    landmarks: list[int]
    pose_valid: list[int]


class ModelMetadata(BaseModel):
    checkpoint: str
    epoch: Optional[int]
    device: str
    vocab_size: int


class SafeguardsInfo(BaseModel):
    max_len: int
    block_ngram_repeat: int
    note: str


class PredictResponse(BaseModel):
    translation: str = Field(..., description="Generated English translation")
    tokens: list[int] = Field(..., description="Decoded vocabulary token IDs")
    duration_ms: float = Field(..., description="Total end-to-end processing duration in ms")
    stages_ms: StagesMs
    feature_shapes: FeatureShapes
    model_metadata: ModelMetadata
    safeguards: SafeguardsInfo


class DemoGalleryItem(BaseModel):
    demo_id: str
    video_id: str
    video_filename: str
    split: str
    category: str
    duration_seconds: Optional[float]
    reference_translation: str
    curated_translation: str
    curated_translation_source: str
    is_curated: bool
    notes: str
    video_url: str


class DemoPredictResponse(BaseModel):
    demo_id: str
    video_id: str
    video_filename: str
    split: str
    category: str
    duration_seconds: Optional[float]
    reference_translation: str
    curated_translation: str
    curated_translation_source: str
    curated_explanation: str
    model_prediction: Optional[str] = None
    model_prediction_source: str
    inference_status: str
    model_metadata: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    stages_ms: Optional[Dict[str, Any]] = None
    tokens: Optional[list[int]] = None
    error_details: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report service health, model readiness, and active execution parameters."""
    if pipeline is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name="ISLMultimodalModel",
            checkpoint=str(DEFAULT_CHECKPOINT),
            checkpoint_epoch=None,
            checkpoint_val_loss=None,
            device="none",
            vocab_size=0,
            max_upload_size_mb=MAX_UPLOAD_SIZE_MB,
            allowed_extensions=sorted(list(ALLOWED_VIDEO_EXTENSIONS)),
        )

    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_name="ISLMultimodalModel",
        checkpoint=str(pipeline.checkpoint_path.name),
        checkpoint_epoch=pipeline.checkpoint_epoch,
        checkpoint_val_loss=pipeline.checkpoint_val_loss,
        device=str(pipeline.device),
        vocab_size=len(pipeline.vocab.itos),
        max_upload_size_mb=MAX_UPLOAD_SIZE_MB,
        allowed_extensions=sorted(list(ALLOWED_VIDEO_EXTENSIONS)),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_translation(
    file: UploadFile = File(..., description="Uploaded video file of ISL signing"),
    max_len: int = Form(
        DEFAULT_MAX_LEN,
        description="Maximum generated tokens before stopping (safeguard).",
    ),
    block_ngram_repeat: int = Form(
        DEFAULT_BLOCK_NGRAM,
        description="N-gram repetition blocking window (default=3 to block repetitive looping).",
    ),
) -> PredictResponse:
    """Accept an uploaded ISL video, extract multimodal features, and return the translation."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not initialized or still loading.",
        )

    # 1. Validate File Extension
    filename = file.filename or "unknown.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed video formats are: {sorted(list(ALLOWED_VIDEO_EXTENSIONS))}"
            ),
        )

    # 2. Stream to Secure Temporary File & Enforce Size Limit
    tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = Path(tmp_file.name)
    total_bytes = 0

    try:
        chunk_size = 1024 * 64  # 64 KB
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                tmp_file.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Uploaded video exceeds maximum allowed limit of "
                        f"{MAX_UPLOAD_SIZE_MB} MB ({MAX_UPLOAD_BYTES} bytes)."
                    ),
                )
            tmp_file.write(chunk)
        tmp_file.flush()
        tmp_file.close()

        if total_bytes == 0:
            raise HTTPException(
                status_code=422,
                detail="Uploaded video file is empty (0 bytes).",
            )

        # 3. Validate Video Stream using OpenCV
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            cap.release()
            raise HTTPException(
                status_code=422,
                detail="Uploaded file could not be parsed as a valid video container.",
            )
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count < 1:
            raise HTTPException(
                status_code=422,
                detail=f"Video stream contains no readable frames (frame_count={frame_count}).",
            )

        # 4. Run Full Inference Pipeline
        try:
            result = pipeline.predict_video(
                video_path=tmp_path,
                max_len=max_len,
                block_ngram_repeat=block_ngram_repeat,
            )
            return PredictResponse(**result)
        except Exception as e:
            logger.error("Inference execution failure for video %s: %s", filename, e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference execution failed: {str(e)}",
            )

    finally:
        # 5. Guaranteed cleanup of temporary video file
        tmp_path.unlink(missing_ok=True)


# ============================================================================
# Demonstration Gallery & Faculty Evaluation Endpoints
# ============================================================================


@app.get("/demo/gallery", response_model=list[DemoGalleryItem])
async def get_demo_gallery() -> list[DemoGalleryItem]:
    """Return curated demonstration gallery with references and safe video streaming URLs."""
    return demo_service.get_all_entries()


@app.get("/demo/video/{demo_id}")
async def get_demo_video(demo_id: str):
    """Stream a curated video from the allowlist with HTTP Range support."""
    try:
        video_path = demo_service.resolve_video_path(demo_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo ID '{demo_id}' not found in gallery allowlist.",
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file for demo '{demo_id}' not found on server disk.",
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_path.name,
    )


@app.post("/demo/predict/{demo_id}", response_model=DemoPredictResponse)
async def predict_demo(
    demo_id: str,
    run_inference: bool = Query(
        False,
        description=(
            "If False, returns human-curated demonstration without model inference. "
            "If True, executes live inference on the video."
        ),
    ),
    max_len: int = Query(
        DEFAULT_MAX_LEN,
        description="Maximum tokens generated before stopping.",
    ),
    block_ngram_repeat: int = Query(
        DEFAULT_BLOCK_NGRAM,
        description="N-gram repetition blocking window.",
    ),
) -> DemoPredictResponse:
    """Retrieve demonstration prediction with clear labelling.

    Clearly distinguishes:
    1. Actual reference translation from dataset.
    2. Human-curated demonstration translation.
    3. Real model prediction (if requested and successful).
    """
    try:
        result = await demo_service.get_demo_prediction(
            demo_id=demo_id,
            run_inference=run_inference,
            pipeline=pipeline,
            max_len=max_len,
            block_ngram_repeat=block_ngram_repeat,
        )
        return DemoPredictResponse(**result)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo ID '{demo_id}' not found in gallery allowlist.",
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file for demo '{demo_id}' not found on disk.",
        )


@app.get("/performance")
async def get_performance_report() -> Dict[str, Any]:
    """Consolidated summary of verified model evaluation metrics across splits and experiments."""
    return demo_service.get_performance_summary()
