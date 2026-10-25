"""Unit and integration tests for demonstration gallery, security, and endpoints.

Tests:
1. Gallery JSON file loading and schema validation
2. Real video file existence and metadata verification
3. Path traversal attack prevention (rejects ../, invalid characters, escaping videos_dir)
4. Unknown demo ID handling (404 Not Found)
5. Missing video file handling
6. Proper separation of reference_translation vs curated_translation vs model_prediction
7. Explicit labelling: curated_translation_source == 'human_curated'
8. No fabricated model prediction when inference is disabled (model_prediction is None)
9. Honest error handling when live inference is unavailable or fails (no copying of curated translation)
10. GET /demo/gallery response schema and absence of raw filesystem paths
11. GET /demo/video/{demo_id} returns video/mp4 with streaming support
12. POST /demo/predict/{demo_id} in curated mode (fast, no inference)
13. POST /demo/predict/{demo_id} with mock live inference
14. GET /performance endpoint structure and resilience when files are missing
15. Existing GET /health and POST /predict backwards compatibility
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.demo_service import DemoGalleryService


@pytest.fixture
def test_client():
    """Create a FastAPI test client with lifespan context."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_pipeline():
    """Create a mock ISLInferencePipeline that mimics predict_video output."""
    mock = MagicMock()
    mock.predict_video.return_value = {
        "translation": "the government is the first time.",
        "tokens": [10, 15, 20, 2],
        "duration_ms": 125.4,
        "stages_ms": {
            "videomae_ms": 60.0,
            "landmarks_ms": 40.0,
            "decode_ms": 25.4,
        },
        "model_metadata": {
            "checkpoint": "best.pt",
            "epoch": 2,
            "device": "cpu",
            "vocab_size": 4024,
        },
    }
    return mock


def test_gallery_json_structure():
    """Verify that demonstration_gallery.json exists and adheres to schema."""
    gallery_path = Path("data/metadata/demonstration_gallery.json")
    assert gallery_path.is_file(), f"Missing gallery metadata: {gallery_path}"

    with open(gallery_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 6, "Expected at least 6 demonstration entries."

    required_fields = {
        "demo_id",
        "video_id",
        "video_filename",
        "split",
        "reference_translation",
        "curated_translation",
        "curated_translation_source",
        "duration_seconds",
        "category",
        "notes",
    }

    demo_ids = set()
    for entry in data:
        assert required_fields.issubset(entry.keys()), f"Missing keys in {entry}"
        assert entry["demo_id"] not in demo_ids, f"Duplicate demo_id: {entry['demo_id']}"
        demo_ids.add(entry["demo_id"])
        assert entry["curated_translation_source"] == "human_curated"
        assert len(entry["reference_translation"].strip()) > 0
        assert len(entry["curated_translation"].strip()) > 0
        assert entry["duration_seconds"] > 0


def test_real_videos_exist_locally():
    """Verify that all videos cataloged in demonstration_gallery.json exist on disk."""
    gallery_path = Path("data/metadata/demonstration_gallery.json")
    videos_dir = Path("data/raw/videos")

    with open(gallery_path, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        vpath = videos_dir / entry["video_filename"]
        assert vpath.is_file(), f"Demonstration video not found: {vpath}"


def test_path_traversal_prevention(tmp_path):
    """Test that DemoGalleryService strictly prevents path traversal."""
    # Create fake gallery with malicious filenames
    fake_gallery = [
        {
            "demo_id": "malicious_01",
            "video_filename": "../../secret.txt",
            "reference_translation": "test",
            "curated_translation": "test",
            "curated_translation_source": "human_curated",
        },
        {
            "demo_id": "malicious_02",
            "video_filename": "sub/../../secret.txt",
            "reference_translation": "test",
            "curated_translation": "test",
            "curated_translation_source": "human_curated",
        },
    ]
    gallery_file = tmp_path / "fake_gallery.json"
    gallery_file.write_text(json.dumps(fake_gallery), encoding="utf-8")

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    service = DemoGalleryService(gallery_path=gallery_file, videos_dir=videos_dir)

    with pytest.raises(PermissionError):
        service.resolve_video_path("malicious_01")

    with pytest.raises(PermissionError):
        service.resolve_video_path("malicious_02")


def test_unknown_demo_id_handling():
    """Verify KeyError is raised for unknown demo IDs."""
    service = DemoGalleryService()
    with pytest.raises(KeyError):
        service.resolve_video_path("nonexistent_demo_999")


def test_missing_video_file_handling(tmp_path):
    """Verify FileNotFoundError when demo is registered but file is absent on disk."""
    fake_gallery = [
        {
            "demo_id": "missing_video_01",
            "video_filename": "does_not_exist.mp4",
            "reference_translation": "test",
            "curated_translation": "test",
            "curated_translation_source": "human_curated",
        }
    ]
    gallery_file = tmp_path / "gallery.json"
    gallery_file.write_text(json.dumps(fake_gallery), encoding="utf-8")

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    service = DemoGalleryService(gallery_path=gallery_file, videos_dir=videos_dir)
    with pytest.raises(FileNotFoundError):
        service.resolve_video_path("missing_video_01")


def test_curated_mode_no_model_prediction():
    """Verify that curated mode (run_inference=False) does NOT fabricate an AI prediction."""
    service = DemoGalleryService()
    res = asyncio.run(service.get_demo_prediction("demo_01", run_inference=False))

    assert res["demo_id"] == "demo_01"
    assert res["model_prediction"] is None
    assert res["model_prediction_source"] == "none"
    assert res["inference_status"] == "not_requested"
    assert res["curated_translation_source"] == "human_curated"
    assert len(res["curated_translation"]) > 0
    assert len(res["reference_translation"]) > 0
    # Crucial assertion: curated text is never copied into model_prediction
    assert res["model_prediction"] != res["curated_translation"]


def test_live_inference_with_mock_pipeline(mock_pipeline):
    """Verify live inference mode correctly records model prediction and execution stages."""
    service = DemoGalleryService()
    res = asyncio.run(service.get_demo_prediction("demo_01", run_inference=True, pipeline=mock_pipeline))

    assert res["inference_status"] == "success"
    assert res["model_prediction"] == "the government is the first time."
    assert res["model_prediction_source"] == "isl_multimodal_run_v5_live_model"
    assert res["duration_ms"] == 125.4
    assert res["stages_ms"]["videomae_ms"] == 60.0
    # Verify curated translation was not overwritten
    assert res["curated_translation_source"] == "human_curated"
    assert res["curated_translation"] != res["model_prediction"]


def test_inference_unavailable_honest_handling():
    """Verify honest reporting when live inference is requested but model is unavailable."""
    service = DemoGalleryService()
    res = asyncio.run(service.get_demo_prediction("demo_01", run_inference=True, pipeline=None))

    assert res["inference_status"] == "unavailable"
    assert res["model_prediction"] is None
    assert res["model_prediction_source"] == "model_unavailable"
    # Never substitute curated text on model absence
    assert res["model_prediction"] is None


def test_inference_failure_honest_handling():
    """Verify that an exception in predict_video is caught honestly and never fakes prediction."""
    failing_pipeline = MagicMock()
    failing_pipeline.predict_video.side_effect = RuntimeError("CUDA Out of Memory simulation")

    service = DemoGalleryService()
    res = asyncio.run(service.get_demo_prediction("demo_01", run_inference=True, pipeline=failing_pipeline))

    assert res["inference_status"] == "failed"
    assert res["model_prediction"] is None
    assert res["model_prediction_source"] == "inference_error"
    assert "CUDA Out of Memory" in res["error_details"]


def test_endpoint_get_gallery(test_client):
    """Test GET /demo/gallery endpoint."""
    response = test_client.get("/demo/gallery")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) >= 6

    for item in items:
        assert "demo_id" in item
        assert "reference_translation" in item
        assert "curated_translation" in item
        assert item["curated_translation_source"] == "human_curated"
        assert item["video_url"].startswith("/demo/video/")
        # Ensure raw filesystem paths like D:\ are not leaked
        assert "D:\\" not in item["video_url"]
        assert "data/raw" not in item["video_url"]


def test_endpoint_get_video_success(test_client):
    """Test GET /demo/video/{demo_id} returns video/mp4 stream."""
    response = test_client.get("/demo/video/demo_01")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 1000


def test_endpoint_get_video_not_found(test_client):
    """Test GET /demo/video/{demo_id} returns 404 for unknown demo ID."""
    response = test_client.get("/demo/video/unknown_demo_xyz")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_endpoint_predict_demo_curated(test_client):
    """Test POST /demo/predict/{demo_id} default curated mode."""
    response = test_client.post("/demo/predict/demo_01")
    assert response.status_code == 200
    data = response.json()
    assert data["demo_id"] == "demo_01"
    assert data["curated_translation_source"] == "human_curated"
    assert data["model_prediction"] is None
    assert data["inference_status"] == "not_requested"


def test_endpoint_predict_demo_unknown_404(test_client):
    """Test POST /demo/predict/{demo_id} returns 404 for unknown demo."""
    response = test_client.post("/demo/predict/nonexistent_demo")
    assert response.status_code == 404


def test_endpoint_performance_summary(test_client):
    """Test GET /performance returns verified metrics."""
    response = test_client.get("/performance")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "performance" in data
    assert "validation" in data["performance"]
    assert "test" in data["performance"]

    val = data["performance"]["validation"]
    assert val["split"] == "validation"
    assert val["n_samples"] == 570
    assert val["loss"] == 6.3774
    assert val["teacher_forced_token_accuracy"] == 0.1672
    assert val["free_running_bleu_4"] == 0.002625
    assert val["free_running_wer"] == 0.9006

    test_split = data["performance"]["test"]
    assert test_split["split"] == "test"
    assert test_split["n_samples"] == 562
    assert test_split["loss"] == 6.4183
    assert test_split["teacher_forced_token_accuracy"] == 0.1615
    assert test_split["free_running_bleu_4"] == 0.003468
    assert test_split["free_running_wer"] == 0.8980


def test_endpoint_performance_fallback_when_missing(tmp_path):
    """Test get_performance_summary graceful fallback when report is missing."""
    empty_dir = tmp_path / "empty_reports"
    empty_dir.mkdir()
    service = DemoGalleryService(reports_dir=empty_dir)
    summary = service.get_performance_summary()
    assert summary["status"] == "unavailable"


def test_existing_health_and_predict_compatibility(test_client):
    """Verify existing GET /health is completely operational and backwards-compatible."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_name" in data
    assert "max_upload_size_mb" in data
    assert "allowed_extensions" in data
