"""MediaPipe landmark extraction for ISL translation.

Extracts pose (33 landmarks) and hand (21 left + 21 right landmarks) keypoints
from sampled video frames using the MediaPipe Tasks API (mediapipe >= 1.0).

Landmark representation
-----------------------
For each sampled frame, 75 landmarks are extracted in the order:
    pose (33) → left hand (21) → right hand (21)

Each landmark contributes (x, y, z) normalised coordinates, giving:
    (33 + 21 + 21) × 3 = 225 values per frame.

Coordinate conventions
-----------------------
MediaPipe's x and y values are normalized to the image dimensions (origin at
top-left; x increases right and y increases down). This extractor does not
normalize hand coordinates relative to a hand bounding box. The z coordinate
is MediaPipe's model-relative depth value; it is not constrained to [0, 1].
No additional mirroring, body-relative normalization, or scaling is applied.
Consumers should use validity masks before interpreting zero-filled detections.

Hand laterality
---------------
MediaPipe reports handedness as detected *in the image* (i.e. the signer's
perspective is already accounted for by MediaPipe's model).  For a front-facing
signer this means "Left" corresponds to the signer's left hand.  The category
name field of the handedness container is "Left" or "Right".  Each frame's
result is mapped into the fixed left-hand and right-hand slots regardless of
the order in which MediaPipe returns the detected hands.

Missing landmarks
-----------------
When pose or a hand is not detected in a frame, the corresponding 63 or 63 (or
21×3=63) coordinate values are filled with 0.0, and the matching validity-mask
slot is set to False.  Callers must check the masks before treating coordinates
as meaningful observations.

Cache file format
-----------------
Each .pt file is a dict with:
    "landmarks"       : float32 tensor [num_frames, 225]
    "pose_valid"      : bool tensor    [num_frames]         True if pose detected
    "left_valid"      : bool tensor    [num_frames]         True if left hand detected
    "right_valid"     : bool tensor    [num_frames]         True if right hand detected
    "source_path"     : str            resolved video path
    "num_frames"      : int            number of sampled frames
    "frame_indices"   : list[int]      frame indices used for sampling
    "representation"  : str            "pose33_lhand21_rhand21_xyz" (version tag)
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_POSE = 33
N_HAND = 21
N_LANDMARKS = N_POSE + N_HAND + N_HAND          # 75
COORDS_PER_LM = 3                                # x, y, z
FEATURE_DIM = N_LANDMARKS * COORDS_PER_LM        # 225
REPRESENTATION_TAG = "pose33_lhand21_rhand21_xyz"

DEFAULT_POSE_MODEL = Path("D:/AI_Cache/mediapipe/pose_landmarker_lite.task")
DEFAULT_HAND_MODEL = Path("D:/AI_Cache/mediapipe/hand_landmarker.task")

_ZERO_POSE = np.zeros(N_POSE * COORDS_PER_LM, dtype=np.float32)   # 99
_ZERO_HAND = np.zeros(N_HAND * COORDS_PER_LM, dtype=np.float32)   # 63


# ---------------------------------------------------------------------------
# Cache key (mirrors feature_extractor.cache_key so keys are the same)
# ---------------------------------------------------------------------------

def cache_key(video_path: str | Path) -> str:
    """Deterministic SHA-1 hash of the resolved absolute video path."""
    return hashlib.sha1(str(Path(video_path).resolve()).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helper: flatten landmark list to 1-D float32 array
# ---------------------------------------------------------------------------

def _sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Return deterministic, ordered sample indices of fixed requested length.

    Short videos may yield repeated indices; retaining repeats preserves the
    model's fixed frame-count contract. VIDEO-mode timestamps are made strictly
    increasing separately when the sampled frames are processed.
    """
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()


def _lm_to_array(landmarks, n_expected: int) -> np.ndarray:
    """Flatten a list of NormalizedLandmark objects into [n_expected * 3] float32.

    Returns a zero array if landmarks is None / empty / wrong length.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(n_expected * COORDS_PER_LM, dtype=np.float32)
    arr = np.empty(n_expected * COORDS_PER_LM, dtype=np.float32)
    for i, lm in enumerate(landmarks[:n_expected]):
        base = i * COORDS_PER_LM
        arr[base] = lm.x if lm.x is not None else 0.0
        arr[base + 1] = lm.y if lm.y is not None else 0.0
        arr[base + 2] = lm.z if lm.z is not None else 0.0
    # Pad if fewer landmarks than expected (should not happen with a valid model)
    actual = min(len(landmarks), n_expected)
    if actual < n_expected:
        arr[actual * COORDS_PER_LM:] = 0.0
    return arr


# ---------------------------------------------------------------------------
# Core extractor class
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """Extract pose + hand landmarks from video frames using MediaPipe Tasks API.

    Parameters
    ----------
    pose_model_path:
        Path to the PoseLandmarker .task file.
    hand_model_path:
        Path to the HandLandmarker .task file.
    num_frames:
        Number of frames to sample uniformly from each video.
    min_pose_detection_confidence:
        Minimum confidence for pose detection.
    min_pose_presence_confidence:
        Minimum confidence for pose presence (tracking).
    min_tracking_confidence:
        Minimum tracking confidence for both landmarkers.
    min_hand_detection_confidence:
        Minimum confidence for hand detection.
    min_hand_presence_confidence:
        Minimum confidence for hand presence.
    """

    def __init__(
        self,
        pose_model_path: str | Path = DEFAULT_POSE_MODEL,
        hand_model_path: str | Path = DEFAULT_HAND_MODEL,
        num_frames: int = 16,
        min_pose_detection_confidence: float = 0.3,
        min_pose_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        min_hand_detection_confidence: float = 0.3,
        min_hand_presence_confidence: float = 0.3,
    ) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        pose_model_path = Path(pose_model_path)
        hand_model_path = Path(hand_model_path)
        if not pose_model_path.is_file():
            raise FileNotFoundError(f"Pose model not found: {pose_model_path}")
        if not hand_model_path.is_file():
            raise FileNotFoundError(f"Hand model not found: {hand_model_path}")

        self.num_frames = num_frames
        self.feature_dim = FEATURE_DIM

        self._pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hand_opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose_lm = None
        self._hand_lm = None
        self._closed = False
        self._init_landmarkers()

    def _init_landmarkers(self) -> None:
        """Initialize fresh landmarker instances to ensure clean state and determinism."""
        self._close_landmarkers()
        self._pose_lm = PoseLandmarker.create_from_options(self._pose_opts)
        self._hand_lm = HandLandmarker.create_from_options(self._hand_opts)

    def _close_landmarkers(self) -> None:
        """Close landmarker instances if open."""
        if self._pose_lm is not None:
            try:
                self._pose_lm.close()
            except Exception:
                pass
            self._pose_lm = None
        if self._hand_lm is not None:
            try:
                self._hand_lm.close()
            except Exception:
                pass
            self._hand_lm = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, video_path: str | Path) -> dict:
        """Extract landmark features from a video file.

        Parameters
        ----------
        video_path:
            Path to an MP4 (or any OpenCV-readable) video file.

        Returns
        -------
        dict with keys:
            "landmarks"    : float32 tensor [num_frames, 225]
            "pose_valid"   : bool tensor    [num_frames]
            "left_valid"   : bool tensor    [num_frames]
            "right_valid"  : bool tensor    [num_frames]
            "source_path"  : str
            "num_frames"   : int
            "frame_indices": list[int]
            "representation": str
        """
        if self._closed:
            raise RuntimeError("LandmarkExtractor has been closed; create a new instance.")

        # Ensure a clean tracking state and reset timestamps for each video
        self._init_landmarkers()

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            if total_frames <= 0:
                raise RuntimeError(f"Video has no readable frames: {video_path}")

            # Uniform sampling indices, mirroring feature_extractor.py
            # Keep duplicate indices for short videos so output always has
            # exactly num_frames rows. Repeated samples are deterministic.
            indices: List[int] = _sample_frame_indices(total_frames, self.num_frames)

            all_coords: List[np.ndarray] = []
            pose_valid: List[bool] = []
            left_valid: List[bool] = []
            right_valid: List[bool] = []

            last_ts = -1

            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, bgr_frame = cap.read()
                if not ok or bgr_frame is None:
                    # Unreadable frame -> zero coords, all masks False
                    all_coords.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                    pose_valid.append(False)
                    left_valid.append(False)
                    right_valid.append(False)
                    continue

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_img = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)

                # Monotonically increasing timestamp in milliseconds from video start
                frame_ms = int(round((frame_idx / fps) * 1000.0))
                timestamp_ms = max(frame_ms, last_ts + 1)
                last_ts = timestamp_ms

                pose_result: PoseLandmarkerResult = self._pose_lm.detect_for_video(
                    mp_img, timestamp_ms
                )
                hand_result: HandLandmarkerResult = self._hand_lm.detect_for_video(
                    mp_img, timestamp_ms
                )

                pose_coords, p_valid = self._process_pose(pose_result)
                left_coords, l_valid, right_coords, r_valid = self._process_hands(hand_result)

                frame_vec = np.concatenate([pose_coords, left_coords, right_coords])
                all_coords.append(frame_vec)
                pose_valid.append(p_valid)
                left_valid.append(l_valid)
                right_valid.append(r_valid)

        finally:
            cap.release()

        landmarks_np = np.stack(all_coords, axis=0)   # [num_frames, 225]
        return {
            "landmarks": torch.from_numpy(landmarks_np),
            "pose_valid": torch.tensor(pose_valid, dtype=torch.bool),
            "left_valid": torch.tensor(left_valid, dtype=torch.bool),
            "right_valid": torch.tensor(right_valid, dtype=torch.bool),
            "source_path": str(video_path.resolve()),
            "num_frames": len(indices),
            "frame_indices": indices,
            "representation": REPRESENTATION_TAG,
        }

    def close(self) -> None:
        """Release MediaPipe landmarker resources."""
        if not self._closed:
            self._close_landmarkers()
            self._closed = True

    # Context-manager support
    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_pose(
        result: PoseLandmarkerResult,
    ) -> Tuple[np.ndarray, bool]:
        """Return (pose_coords [99], pose_detected)."""
        if result.pose_landmarks:
            lm_list = result.pose_landmarks[0]  # only one pose requested
            return _lm_to_array(lm_list, N_POSE), True
        return _ZERO_POSE.copy(), False

    @staticmethod
    def _process_hands(
        result: HandLandmarkerResult,
    ) -> Tuple[np.ndarray, bool, np.ndarray, bool]:
        """Return (left_coords [63], left_valid, right_coords [63], right_valid).

        Handedness classification
        ------------------------
        MediaPipe's HandLandmarker reports the handedness of the detected hand
        as seen in the image using the *signer's* perspective (i.e. it already
        accounts for the model's mirror convention).  The category_name of the
        highest-score handedness category is "Left" or "Right" and refers to
        the signer's hand.  We use this label directly to slot each detected
        hand into the fixed left / right output positions.
        """
        left_coords = _ZERO_HAND.copy()
        right_coords = _ZERO_HAND.copy()
        left_detected = False
        right_detected = False

        for handedness_list, lm_list in zip(result.handedness, result.hand_landmarks):
            if not handedness_list:
                continue
            # The best handedness category is the first (highest score)
            side = handedness_list[0].category_name  # "Left" or "Right"
            coords = _lm_to_array(lm_list, N_HAND)
            if side == "Left":
                left_coords = coords
                left_detected = True
            elif side == "Right":
                right_coords = coords
                right_detected = True

        return left_coords, left_detected, right_coords, right_detected


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def validate_cache(
    payload: dict,
    video_path: str | Path | None = None,
    expected_num_frames: int | None = None,
) -> bool:
    """Validate a landmark-cache payload, optionally against its source video.

    The cache schema is deliberately strict so stale or malformed caches are
    not silently reused. ``video_path`` enables source-path verification.
    """
    if not isinstance(payload, dict):
        return False
    required_keys = {
        "landmarks", "pose_valid", "left_valid", "right_valid",
        "source_path", "num_frames", "frame_indices", "representation",
    }
    if not required_keys.issubset(payload.keys()):
        return False

    lm = payload["landmarks"]
    if not isinstance(lm, torch.Tensor):
        return False
    if lm.dtype != torch.float32 or lm.ndim != 2 or lm.shape[1] != FEATURE_DIM:
        return False
    n = int(lm.shape[0])
    if n < 1 or isinstance(payload["num_frames"], bool) or payload["num_frames"] != n:
        return False
    if expected_num_frames is not None and n != expected_num_frames:
        return False
    if not torch.isfinite(lm).all().item():
        return False

    for key in ("pose_valid", "left_valid", "right_valid"):
        mask = payload[key]
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (n,):
            return False

    source_path = payload["source_path"]
    if not isinstance(source_path, str) or not source_path.strip():
        return False
    try:
        if str(Path(source_path).resolve()) != source_path:
            return False
        if video_path is not None and str(Path(video_path).resolve()) != source_path:
            return False
    except (OSError, TypeError, ValueError):
        return False

    if payload["representation"] != REPRESENTATION_TAG:
        return False

    indices = payload["frame_indices"]
    if not isinstance(indices, (list, tuple)) or len(indices) != n:
        return False
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in indices):
        return False
    # Duplicate indices are valid for videos shorter than the requested frame
    # count; ordering must still be monotonic.
    if any(indices[i] > indices[i + 1] for i in range(n - 1)):
        return False
    return True


def landmarks_to_cache(
    video_path: str | Path,
    cache_dir: str | Path,
    extractor: LandmarkExtractor,
    overwrite: bool = False,
) -> Path:
    """Extract landmarks and save to cache atomically.

    Parameters
    ----------
    video_path:
        Source video file.
    cache_dir:
        Directory in which to store .pt cache files.
    extractor:
        An open LandmarkExtractor instance.
    overwrite:
        If False (default) and a valid cache file already exists, skip
        extraction and return the existing path.

    Returns
    -------
    Path to the (new or existing) cache file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = cache_dir / f"{cache_key(video_path)}.pt"

    if out_path.exists() and not overwrite:
        # Validate existing cache before trusting it
        try:
            payload = torch.load(out_path, map_location="cpu", weights_only=False)
            if validate_cache(payload, video_path=video_path, expected_num_frames=extractor.num_frames):
                return out_path
            # Invalid → fall through and re-extract
        except Exception:
            pass

    payload = extractor.extract(video_path)

    # Atomic write: save to a temp file in the same directory, then rename.
    tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return out_path
