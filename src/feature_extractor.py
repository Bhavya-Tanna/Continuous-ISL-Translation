"""ResNet-18 frame feature extraction and per-video caching."""
from pathlib import Path
import hashlib
import cv2
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def cache_key(video_path):
    return hashlib.sha1(str(Path(video_path).resolve()).encode("utf-8")).hexdigest()


class ResNet18FrameExtractor:
    def __init__(self, device=None, num_frames=4, pretrained=True):
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.num_frames = num_frames
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.preprocess = weights.transforms() if weights else ResNet18_Weights.DEFAULT.transforms()
        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.feature_dim = 512

    @torch.inference_mode()
    def extract(self, video_path):
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open video: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {path}")
        indices = sorted(set(torch.linspace(0, n - 1, self.num_frames).round().long().tolist()))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.preprocess(Image.fromarray(rgb)))
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode frames from: {path}")
        feats = self.model(torch.stack(frames).to(self.device)).flatten(1).cpu()
        return feats.contiguous()


def extract_to_cache(video_path, cache_dir, extractor, overwrite=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{cache_key(video_path)}.pt"
    if out.exists() and not overwrite:
        return out
    feats = extractor.extract(video_path)
    torch.save({"features": feats, "source_path": str(Path(video_path)), "num_frames": len(feats)}, out)
    return out
