import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from src.landmark_extractor import (
    COORDS_PER_LM, FEATURE_DIM, N_HAND, N_LANDMARKS, N_POSE,
    REPRESENTATION_TAG, LandmarkExtractor, _lm_to_array, _sample_frame_indices, cache_key, validate_cache,
    landmarks_to_cache,
)


def valid_payload(source_path="/path/to/video.mp4", n_frames=16):
    return {
        "landmarks": torch.zeros((n_frames, FEATURE_DIM), dtype=torch.float32),
        "pose_valid": torch.ones(n_frames, dtype=torch.bool),
        "left_valid": torch.zeros(n_frames, dtype=torch.bool),
        "right_valid": torch.ones(n_frames, dtype=torch.bool),
        "source_path": str(Path(source_path).resolve()),
        "num_frames": n_frames,
        "frame_indices": list(range(n_frames)),
        "representation": REPRESENTATION_TAG,
    }


def test_landmark_constants():
    assert (N_POSE, N_HAND, N_LANDMARKS, COORDS_PER_LM, FEATURE_DIM) == (33, 21, 75, 3, 225)


def test_cache_key_deterministic():
    path_a = "data/raw/videos/test.mp4"
    path_b = str(Path(path_a).resolve())
    assert cache_key(path_a) == cache_key(path_b)
    assert cache_key(path_a) == hashlib.sha1(path_b.encode("utf-8")).hexdigest()


def test_lm_to_array_empty_and_missing_coordinates():
    assert np.all(_lm_to_array([], N_POSE) == 0.0)
    assert np.all(_lm_to_array(None, N_HAND) == 0.0)
    arr = _lm_to_array([SimpleNamespace(x=None, y=0.25, z=-0.2)], 2)
    assert arr.dtype == np.float32
    assert np.allclose(arr, [0.0, 0.25, -0.2, 0.0, 0.0, 0.0])


def test_validate_cache_accepts_valid_payload_and_optional_source_check(tmp_path):
    video = tmp_path / "video.mp4"
    payload = valid_payload(video)
    assert validate_cache(payload)
    assert validate_cache(payload, video_path=video, expected_num_frames=16)
    assert not validate_cache(payload, video_path=tmp_path / "other.mp4")
    assert not validate_cache(payload, expected_num_frames=8)


@pytest.mark.parametrize("field", ["landmarks", "pose_valid", "left_valid", "right_valid", "source_path", "num_frames", "frame_indices", "representation"])
def test_validate_cache_rejects_missing_fields(field):
    payload = valid_payload()
    del payload[field]
    assert not validate_cache(payload)


def test_validate_cache_rejects_bad_shape_dtype_and_nonfinite():
    payload = valid_payload(); payload["landmarks"] = torch.zeros((16, 224), dtype=torch.float32)
    assert not validate_cache(payload)
    payload = valid_payload(); payload["landmarks"] = torch.zeros((16, 225), dtype=torch.float64)
    assert not validate_cache(payload)
    payload = valid_payload(); payload["landmarks"][0, 0] = float("nan")
    assert not validate_cache(payload)
    payload = valid_payload(); payload["landmarks"][0, 0] = float("inf")
    assert not validate_cache(payload)


@pytest.mark.parametrize("field", ["pose_valid", "left_valid", "right_valid"])
def test_validate_cache_rejects_bad_masks(field):
    payload = valid_payload(); payload[field] = torch.ones(16, dtype=torch.uint8)
    assert not validate_cache(payload)
    payload = valid_payload(); payload[field] = torch.ones(15, dtype=torch.bool)
    assert not validate_cache(payload)


def test_validate_cache_rejects_wrong_tag_and_inconsistent_indices():
    payload = valid_payload(); payload["representation"] = "wrong_tag"
    assert not validate_cache(payload)
    payload = valid_payload(); payload["frame_indices"] = list(range(15))
    assert not validate_cache(payload)
    payload = valid_payload(); payload["frame_indices"] = list(range(15, -1, -1))
    assert not validate_cache(payload)
    payload = valid_payload(); payload["num_frames"] = 15
    assert not validate_cache(payload)


def test_validate_cache_allows_duplicate_indices_for_short_video():
    payload = valid_payload(n_frames=16)
    payload["frame_indices"] = [0] * 8 + list(range(1, 9))
    assert validate_cache(payload)


def test_cache_reuse_checks_source_and_schema(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    cache_dir = tmp_path / "cache"; cache_dir.mkdir()
    out_path = cache_dir / f"{cache_key(video)}.pt"
    torch.save(valid_payload(video), out_path)

    class NeverExtract:
        num_frames = 16
        def extract(self, path):
            raise AssertionError("valid cache should be reused without extraction")
    assert landmarks_to_cache(video, cache_dir, NeverExtract(), overwrite=False) == out_path

    payload = valid_payload(tmp_path / "other.mp4")
    torch.save(payload, out_path)
    class FakeExtractor:
        num_frames = 16
        def extract(self, path):
            return valid_payload(path)
    landmarks_to_cache(video, cache_dir, FakeExtractor(), overwrite=False)
    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert reloaded["source_path"] == str(video.resolve())


def test_short_video_sampling_keeps_fixed_length_and_is_deterministic():
    first = _sample_frame_indices(total_frames=3, num_frames=16)
    second = _sample_frame_indices(total_frames=3, num_frames=16)
    assert first == second
    assert len(first) == 16
    assert first[0] == 0 and first[-1] == 2
    assert first == sorted(first)
    assert set(first) == {0, 1, 2}


def test_sampling_rejects_empty_video():
    with pytest.raises(ValueError):
        _sample_frame_indices(total_frames=0, num_frames=16)


def test_missing_pose_and_hands_return_zero_coordinates_and_false_masks():
    pose_coords, pose_valid = LandmarkExtractor._process_pose(SimpleNamespace(pose_landmarks=[]))
    left_coords, left_valid, right_coords, right_valid = LandmarkExtractor._process_hands(
        SimpleNamespace(handedness=[], hand_landmarks=[])
    )
    assert pose_coords.shape == (99,) and not pose_valid and np.all(pose_coords == 0)
    assert left_coords.shape == (63,) and not left_valid and np.all(left_coords == 0)
    assert right_coords.shape == (63,) and not right_valid and np.all(right_coords == 0)
