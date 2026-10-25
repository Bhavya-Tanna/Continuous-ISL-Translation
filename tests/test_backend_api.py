"""Tests for FastAPI backend and multimodal inference pipeline.

Covers:
- Checkpoint and vocabulary loading
- Feature tensor shapes ([1, 16, 768] and [1, 16, 225])
- N-gram repetition blocking logic
- GET /health endpoint
- POST /predict validation (invalid extension, file too large, corrupt video)
- POST /predict end-to-end with real video file
"""

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
from fastapi.testclient import TestClient

from src.api import app, MAX_UPLOAD_SIZE_MB
from src.inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_VOCAB,
    ISLInferencePipeline,
    decode_autoregressive,
)
from src.multimodal_model import ISLMultimodalModel
from src.text import Vocabulary

REAL_TEST_VIDEO = Path("data/raw/videos/-CpDPjXeB2c--1.mp4")


def test_checkpoint_and_vocab_loading():
    """Verify best.pt checkpoint loads cleanly with expected architecture."""
    assert DEFAULT_CHECKPOINT.is_file(), f"Missing checkpoint: {DEFAULT_CHECKPOINT}"
    assert DEFAULT_CONFIG.is_file(), f"Missing config: {DEFAULT_CONFIG}"
    assert DEFAULT_VOCAB.is_file(), f"Missing vocab: {DEFAULT_VOCAB}"

    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        config = json.load(f)
    vocab = Vocabulary.load(DEFAULT_VOCAB)

    assert len(vocab.itos) == 4024
    assert config["rgb_dim"] == 768
    assert config["landmark_dim"] == 225

    model = ISLMultimodalModel(
        vocab_size=len(vocab.itos),
        rgb_dim=config["rgb_dim"],
        landmark_dim=config["landmark_dim"],
        fusion_dim=config["fusion_dim"],
        encoder_hidden=config["encoder_hidden"],
        decoder_hidden=config["decoder_hidden"],
        embedding_dim=config["embedding_dim"],
        attention_dim=config["attention_dim"],
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
    )
    blob = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    assert "model" in blob
    model.load_state_dict(blob["model"])
    model.eval()


def test_ngram_repetition_blocking_unit():
    """Verify that n-gram repetition blocking suppresses repeated sequences."""
    vocab = Vocabulary.load(DEFAULT_VOCAB)
    # Instantiate a tiny dummy model with pad=0, bos=1, eos=2
    model = ISLMultimodalModel(
        vocab_size=len(vocab.itos),
        rgb_dim=768,
        landmark_dim=225,
        fusion_dim=64,
        encoder_hidden=32,
        decoder_hidden=64,
        embedding_dim=32,
        attention_dim=32,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
    ).eval()

    rgb = torch.randn(1, 16, 768)
    lm = torch.zeros(1, 16, 225)
    pv = torch.ones(1, 16, dtype=torch.bool)
    lv = torch.ones(1, 16, dtype=torch.bool)
    rv = torch.ones(1, 16, dtype=torch.bool)
    mask = torch.ones(1, 16, dtype=torch.bool)

    # Decode with block_ngram_repeat=3 and max_len=15
    text, tokens = decode_autoregressive(
        model=model,
        rgb_features=rgb,
        landmarks=lm,
        pose_valid=pv,
        left_valid=lv,
        right_valid=rv,
        feature_mask=mask,
        vocab=vocab,
        max_len=15,
        block_ngram_repeat=3,
    )

    # Check that no 3-gram appears more than once in the output tokens
    if len(tokens) >= 3:
        trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        assert len(trigrams) == len(set(trigrams)), f"Duplicate 3-gram found: {trigrams}"


@pytest.fixture(scope="module")
def client():
    """Reusable TestClient running the FastAPI app with loaded pipeline."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_endpoint(client):
    """GET /health must return HTTP 200 with status 'ready' and valid metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_name"] == "ISLMultimodalModel"
    assert data["checkpoint"] == "best.pt"
    assert data["checkpoint_epoch"] == 2
    assert data["vocab_size"] == 4024
    assert ".mp4" in data["allowed_extensions"]


def test_predict_invalid_extension(client):
    """POST /predict must reject non-video files with HTTP 400."""
    fake_file = io.BytesIO(b"Hello plain text")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_predict_file_too_large(client, monkeypatch):
    """POST /predict must reject files exceeding upload size limit with HTTP 413."""
    # Temporarily set max upload to 1 KB to test size enforcement without sending 50MB in test
    import src.api as api_mod
    monkeypatch.setattr(api_mod, "MAX_UPLOAD_BYTES", 1024)

    oversized_data = io.BytesIO(b"0" * 2048)
    response = client.post(
        "/predict",
        files={"file": ("large.mp4", oversized_data, "video/mp4")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]


def test_predict_corrupt_video(client):
    """POST /predict with corrupted/unreadable video bytes must return HTTP 422."""
    corrupt_bytes = io.BytesIO(b"\x00\x01\x02\x03\x04\x05junkdata")
    response = client.post(
        "/predict",
        files={"file": ("corrupt.mp4", corrupt_bytes, "video/mp4")},
    )
    assert response.status_code == 422
    assert "could not be parsed" in response.json()["detail"]


def test_predict_real_video(client):
    """POST /predict with a real video file must succeed end-to-end (HTTP 200)."""
    assert REAL_TEST_VIDEO.is_file(), f"Missing real test video: {REAL_TEST_VIDEO}"

    with open(REAL_TEST_VIDEO, "rb") as f:
        video_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": (REAL_TEST_VIDEO.name, io.BytesIO(video_bytes), "video/mp4")},
        data={"max_len": 25, "block_ngram_repeat": 3},
    )

    assert response.status_code == 200, f"Predict failed: {response.text}"
    data = response.json()

    # Verify response schema fields
    assert "translation" in data
    assert isinstance(data["translation"], str)
    assert len(data["translation"]) > 0
    assert "duration_ms" in data
    assert data["duration_ms"] > 0
    assert "stages_ms" in data
    assert "videomae_ms" in data["stages_ms"]
    assert "landmarks_ms" in data["stages_ms"]
    assert "decode_ms" in data["stages_ms"]

    # Verify tensor shapes
    assert data["feature_shapes"]["rgb_features"] == [1, 16, 768]
    assert data["feature_shapes"]["landmarks"] == [1, 16, 225]

    # Verify safeguards metadata
    assert data["safeguards"]["max_len"] == 25
    assert data["safeguards"]["block_ngram_repeat"] == 3
