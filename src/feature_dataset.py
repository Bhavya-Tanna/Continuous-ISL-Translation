"""Load cached visual features for manifest rows; pair/triple windows concatenate in order."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .dataset_loader import ISLDataset
from .feature_extractor import cache_key


class ISLFeatureDataset(Dataset):
    def __init__(self, manifest_path, split, video_dir, feature_dir, window_type=None, check_files=True):
        self.rows = ISLDataset(manifest_path, split, video_dir, window_type, check_files)
        self.feature_dir = Path(feature_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        parts = []
        for video_path in row["video_paths"]:
            cache_path = self.feature_dir / f"{cache_key(video_path)}.pt"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing feature cache {cache_path}; run scripts/extract_features.py")
            saved = torch.load(cache_path, map_location="cpu", weights_only=False)
            feats = saved["features"] if isinstance(saved, dict) else saved
            if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
                raise ValueError(f"Expected [frames, feature_dim] tensor in {cache_path}")
            parts.append(feats.float())
        return {
            "features": torch.cat(parts, dim=0),
            "target_text": row["target_text"],
            "video_id": row["video_id"],
            "window_type": row["window_type"],
            "split": row["split"],
        }
