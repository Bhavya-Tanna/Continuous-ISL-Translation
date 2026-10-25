"""Demonstration gallery and evaluation service for Continuous ISL Translation.

Manages curated demonstration clips, secure video streaming paths,
labelled demonstration responses (distinguishing dataset references,
human-curated translations, and actual model predictions), and consolidated
performance reports.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("isl.demo_service")

# Default paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GALLERY_PATH = BASE_DIR / "data" / "metadata" / "demonstration_gallery.json"
DEFAULT_VIDEOS_DIR = BASE_DIR / "data" / "raw" / "videos"
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"

# Global inference lock to prevent GPU concurrency hazards
_INFERENCE_LOCK = asyncio.Lock()


class DemoGalleryService:
    """Service handling curated demonstration gallery access, security, and inference."""

    def __init__(
        self,
        gallery_path: Path | str = DEFAULT_GALLERY_PATH,
        videos_dir: Path | str = DEFAULT_VIDEOS_DIR,
        reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    ) -> None:
        self.gallery_path = Path(gallery_path)
        self.videos_dir = Path(videos_dir).resolve()
        self.reports_dir = Path(reports_dir)
        self._gallery: Dict[str, Dict[str, Any]] = {}
        self.reload_gallery()

    def reload_gallery(self) -> None:
        """Load or reload the gallery metadata from disk."""
        if not self.gallery_path.is_file():
            logger.warning("Demonstration gallery file not found: %s", self.gallery_path)
            self._gallery = {}
            return

        with open(self.gallery_path, encoding="utf-8") as f:
            data = json.load(f)

        gallery_map = {}
        for entry in data:
            demo_id = entry.get("demo_id")
            if not demo_id:
                continue
            gallery_map[demo_id] = entry

        self._gallery = gallery_map
        logger.info("Loaded %d demonstration entries from %s", len(self._gallery), self.gallery_path)

    def get_all_entries(self, base_url: str = "") -> List[Dict[str, Any]]:
        """Return public gallery metadata with safe video streaming URLs.

        Arbitrary internal filesystem paths are never exposed.
        """
        results = []
        for demo_id, item in self._gallery.items():
            results.append({
                "demo_id": item["demo_id"],
                "video_id": item["video_id"],
                "video_filename": item["video_filename"],
                "split": item.get("split", "unknown"),
                "category": item.get("category", "General"),
                "duration_seconds": item.get("duration_seconds"),
                "reference_translation": item.get("reference_translation", ""),
                "curated_translation": item.get("curated_translation", ""),
                "curated_translation_source": item.get("curated_translation_source", "human_curated"),
                "is_curated": True,
                "notes": item.get("notes", ""),
                "video_url": f"/demo/video/{demo_id}",
            })
        return results

    def get_entry(self, demo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a raw gallery entry by demo_id."""
        return self._gallery.get(demo_id)

    def resolve_video_path(self, demo_id: str) -> Path:
        """Resolve and strictly validate the video path for a demo ID.

        Enforces:
        1. demo_id must exist in the curated allowlist.
        2. Resolved file path must be strictly inside the configured videos_dir.
        3. Target file must actually exist on disk.

        Raises:
            KeyError: if demo_id is not in allowlist.
            PermissionError: if path traversal is attempted.
            FileNotFoundError: if the video file does not exist.
        """
        if demo_id not in self._gallery:
            raise KeyError(f"Demo ID '{demo_id}' is not in the curated gallery allowlist.")

        entry = self._gallery[demo_id]
        filename = entry.get("video_filename", "")
        if not filename or "/" in filename or "\\" in filename:
            raise PermissionError(f"Invalid or unsafe video filename detected for '{demo_id}'.")

        target_path = (self.videos_dir / filename).resolve()

        # Strict path traversal check
        try:
            target_path.relative_to(self.videos_dir)
        except ValueError:
            raise PermissionError(f"Path traversal attempt rejected for demo_id '{demo_id}'.")

        if not target_path.is_file():
            raise FileNotFoundError(f"Video file for '{demo_id}' not found on disk at {target_path}")

        return target_path

    async def get_demo_prediction(
        self,
        demo_id: str,
        run_inference: bool = False,
        pipeline: Optional[Any] = None,
        max_len: int = 30,
        block_ngram_repeat: int = 3,
    ) -> Dict[str, Any]:
        """Produce a demonstration response clearly separating references, curated text, and AI predictions.

        Parameters
        ----------
        demo_id:
            Target demonstration identifier from gallery allowlist.
        run_inference:
            If False, returns curated demonstration without running model inference.
            If True, safely executes model inference using the existing pipeline.
        pipeline:
            The loaded ISLInferencePipeline instance (if available).
        """
        entry = self.get_entry(demo_id)
        if not entry:
            raise KeyError(f"Demo ID '{demo_id}' not found in gallery allowlist.")

        video_path = self.resolve_video_path(demo_id)

        # Baseline response containing reference and curated translation
        response: Dict[str, Any] = {
            "demo_id": demo_id,
            "video_id": entry["video_id"],
            "video_filename": entry["video_filename"],
            "split": entry.get("split", "unknown"),
            "category": entry.get("category", "General"),
            "duration_seconds": entry.get("duration_seconds"),
            "reference_translation": entry.get("reference_translation", ""),
            "curated_translation": entry.get("curated_translation", ""),
            "curated_translation_source": entry.get("curated_translation_source", "human_curated"),
            "curated_explanation": (
                "Human-curated fluent rendering representing the ground truth meaning of the sign sequence."
            ),
            "model_prediction": None,
            "model_prediction_source": "none",
            "inference_status": "not_requested",
            "model_metadata": None,
            "duration_ms": None,
            "stages_ms": None,
        }

        if not run_inference:
            return response

        # Live inference requested
        if pipeline is None:
            response["inference_status"] = "unavailable"
            response["model_prediction_source"] = "model_unavailable"
            logger.warning("Live inference requested for %s, but pipeline is not initialized.", demo_id)
            return response

        # Execute live inference safely under GPU lock
        async with _INFERENCE_LOCK:
            try:
                # Offload heavy model inference to background thread to avoid event loop starvation
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: pipeline.predict_video(
                        video_path=video_path,
                        max_len=max_len,
                        block_ngram_repeat=block_ngram_repeat,
                    ),
                )
                response["model_prediction"] = result.get("translation", "")
                response["model_prediction_source"] = "isl_multimodal_run_v5_live_model"
                response["inference_status"] = "success"
                response["model_metadata"] = result.get("model_metadata")
                response["duration_ms"] = result.get("duration_ms")
                response["stages_ms"] = result.get("stages_ms")
                response["tokens"] = result.get("tokens")
            except Exception as e:
                logger.error("Live inference failed for demo '%s': %s", demo_id, e, exc_info=True)
                response["inference_status"] = "failed"
                response["model_prediction"] = None
                response["model_prediction_source"] = "inference_error"
                response["error_details"] = str(e)

        return response

    def get_performance_summary(self) -> Dict[str, Any]:
        """Aggregate consolidated performance metrics from verified report artifacts."""
        perf_path = self.reports_dir / "final_model_performance.json"
        comp_path = self.reports_dir / "model_comparison.json"

        if not perf_path.is_file():
            # Fallback if reports directory is missing
            return {
                "status": "unavailable",
                "message": f"Performance report file not found at {perf_path}",
            }

        with open(perf_path, encoding="utf-8") as f:
            perf_data = json.load(f)

        comp_data = {}
        if comp_path.is_file():
            with open(comp_path, encoding="utf-8") as f:
                comp_data = json.load(f)

        return {
            "status": "available",
            "model_name": perf_data.get("model_name"),
            "checkpoint_path": perf_data.get("checkpoint_path"),
            "best_epoch": perf_data.get("best_epoch"),
            "architecture_details": perf_data.get("architecture_details"),
            "decoding_configuration": perf_data.get("decoding_configuration"),
            "metrics_definitions": perf_data.get("metrics_definitions"),
            "performance": perf_data.get("performance"),
            "experiments_comparison": comp_data.get("experiments", []),
            "decoding_strategy_comparison": comp_data.get("decoding_strategy_comparison_run_v5", []),
            "assessment": perf_data.get("assessment"),
            "source_artifacts": [
                "reports/final_model_performance.json",
                "reports/model_comparison.json",
            ],
        }
