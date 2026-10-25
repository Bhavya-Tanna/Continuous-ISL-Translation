"""Inference pipeline for Continuous ISL Translation using Multimodal Run v5 (16 steps).

Loads:
- ISLMultimodalModel (trained best.pt from multimodal_run_v5_16steps)
- Vocabulary (4,024 tokens)
- VideoMAE-base (16-step tubelet features from 32 sampled video frames)
- MediaPipe Tasks (Pose 33 + Left Hand 21 + Right Hand 21 = 225 landmarks across 16 frames)

Provides safe autoregressive decoding with configurable n-gram repetition blocking.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEModel

from src.landmark_extractor import (
    DEFAULT_HAND_MODEL,
    DEFAULT_POSE_MODEL,
    LandmarkExtractor,
)
from src.multimodal_model import ISLMultimodalModel
from src.text import Vocabulary

logger = logging.getLogger("isl.inference")

LOCAL_VIDEOMAE_PATH = Path(
    r"D:\AI_Cache\huggingface\hub\models--MCG-NJU--videomae-base\snapshots\dc740ceda42fce44faed2ea03c6d447db72f6af9"
)
DEFAULT_CHECKPOINT = Path(
    "data/processed/multimodal_run_v5_16steps/checkpoints/best.pt"
)
DEFAULT_CONFIG = Path("data/processed/multimodal_run_v5_16steps/config.json")
DEFAULT_VOCAB = Path("data/processed/multimodal_run_v5_16steps/vocab.json")


def read_32_frames_from_video(video_path: Path | str) -> List[Image.Image]:
    """Read 32 evenly spaced RGB frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise RuntimeError(f"No readable frames in video: {video_path}")

        indices = np.linspace(0, frame_count - 1, 32, dtype=int)
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        return frames
    finally:
        cap.release()


@torch.inference_mode()
def decode_autoregressive(
    model: ISLMultimodalModel,
    rgb_features: torch.Tensor,
    landmarks: torch.Tensor,
    pose_valid: torch.Tensor,
    left_valid: torch.Tensor,
    right_valid: torch.Tensor,
    feature_mask: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 30,
    block_ngram_repeat: int = 3,
) -> Tuple[str, List[int]]:
    """Autoregressive decoding with optional n-gram repetition blocking.

    Parameters
    ----------
    model:
        Trained ISLMultimodalModel in eval mode.
    rgb_features:
        [1, 16, 768] VideoMAE feature tensor.
    landmarks:
        [1, 16, 225] MediaPipe landmark coordinates.
    pose_valid, left_valid, right_valid:
        [1, 16] boolean detection masks.
    feature_mask:
        [1, 16] boolean valid-frame mask (all True for complete 16-frame windows).
    vocab:
        Vocabulary instance for decoding token IDs to string.
    max_len:
        Maximum sequence length before stopping.
    block_ngram_repeat:
        If > 0, bans generating the N-th token if the resulting N-gram has already
        appeared earlier in the generated sequence.
    """
    device = rgb_features.device
    fused = model.fuse(rgb_features, landmarks, pose_valid, left_valid, right_valid)
    enc_outputs, hidden = model.decoder.encode(fused, feature_mask)

    token = torch.full((1,), model.decoder.bos_id, dtype=torch.long, device=device)
    outputs: List[int] = []

    for _ in range(max_len):
        emb = model.decoder.embedding(token)
        context, _ = model.decoder.attention(enc_outputs, hidden, feature_mask)
        hidden = model.decoder.decoder(torch.cat([emb, context], dim=-1), hidden)
        out_repr = (
            torch.cat([hidden, context, emb], dim=-1)
            if model.decoder.use_embedding_in_output
            else torch.cat([hidden, context], dim=-1)
        )
        logits = model.decoder.output(out_repr)  # [1, vocab_size]

        # Never predict pad_id or bos_id during generation
        logits[:, model.decoder.pad_id] = -1e9
        logits[:, model.decoder.bos_id] = -1e9

        # N-gram repetition blocking
        if block_ngram_repeat > 0 and len(outputs) >= block_ngram_repeat - 1:
            n_minus_1 = block_ngram_repeat - 1
            prefix = tuple(outputs[-n_minus_1:])
            # Search earlier history for this exact (n-1)-gram
            for i in range(len(outputs) - n_minus_1):
                if tuple(outputs[i : i + n_minus_1]) == prefix:
                    banned_token = outputs[i + n_minus_1]
                    logits[:, banned_token] = -1e9

        token = logits.argmax(-1)
        token_id = int(token.item())

        if token_id == model.decoder.eos_id:
            break

        outputs.append(token_id)

    decoded_text = vocab.decode(outputs)
    return decoded_text, outputs


class ISLInferencePipeline:
    """Production inference pipeline holding models and preprocessing assets."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        config_path: Path | str = DEFAULT_CONFIG,
        vocab_path: Path | str = DEFAULT_VOCAB,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.vocab_path = Path(vocab_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not self.vocab_path.is_file():
            raise FileNotFoundError(f"Vocab not found: {self.vocab_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Initializing ISLInferencePipeline on device: %s", self.device)

        # 1. Load Config & Vocabulary
        with open(self.config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        self.vocab = Vocabulary.load(self.vocab_path)

        # 2. Load ISLMultimodalModel
        self.model = ISLMultimodalModel(
            vocab_size=len(self.vocab.itos),
            rgb_dim=self.config["rgb_dim"],
            landmark_dim=self.config["landmark_dim"],
            fusion_dim=self.config["fusion_dim"],
            encoder_hidden=self.config["encoder_hidden"],
            decoder_hidden=self.config["decoder_hidden"],
            embedding_dim=self.config["embedding_dim"],
            attention_dim=self.config["attention_dim"],
            pad_id=self.vocab.pad_id,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
            dropout=self.config.get("dropout", 0.1),
            use_embedding_in_output=self.config.get("use_embedding_in_output", False),
            word_dropout=self.config.get("word_dropout", 0.0),
            unk_id=self.vocab.unk_id,
        )

        blob = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.checkpoint_epoch = blob.get("epoch", None)
        self.checkpoint_val_loss = blob.get("val_loss", None)
        self.model.load_state_dict(blob["model"])
        self.model.to(self.device).eval()
        logger.info(
            "ISLMultimodalModel loaded from %s (epoch=%s, val_loss=%.4f)",
            self.checkpoint_path,
            self.checkpoint_epoch,
            self.checkpoint_val_loss or 0.0,
        )

        # 3. Load VideoMAE
        model_source = (
            LOCAL_VIDEOMAE_PATH if LOCAL_VIDEOMAE_PATH.exists() else "MCG-NJU/videomae-base"
        )
        logger.info("Loading VideoMAE from: %s", model_source)
        self.videomae_processor = VideoMAEImageProcessor.from_pretrained(model_source)
        self.videomae_model = VideoMAEModel.from_pretrained(model_source)
        self.videomae_model.to(self.device).eval()

        # 4. Load MediaPipe LandmarkExtractor
        self.landmark_extractor = LandmarkExtractor(
            pose_model_path=DEFAULT_POSE_MODEL,
            hand_model_path=DEFAULT_HAND_MODEL,
            num_frames=16,
        )
        logger.info("MediaPipe LandmarkExtractor initialized (16 frames).")

    @torch.inference_mode()
    def extract_videomae_features(self, video_path: Path | str) -> torch.Tensor:
        """Extract [1, 16, 768] VideoMAE features from 32 sampled frames."""
        frames = read_32_frames_from_video(video_path)
        clip1 = frames[:16]
        clip2 = frames[16:]

        inputs1 = self.videomae_processor(clip1, return_tensors="pt")["pixel_values"]
        inputs2 = self.videomae_processor(clip2, return_tensors="pt")["pixel_values"]
        pixel_values = torch.cat([inputs1, inputs2], dim=0).to(self.device)  # [2, 16, 3, 224, 224]

        vmae_out = self.videomae_model(pixel_values=pixel_values).last_hidden_state  # [2, 1568, 768]
        # Pool across spatial patches (196 patches per tubelet) -> [2, 8, 768]
        pooled = vmae_out.reshape(2, 8, 196, 768).mean(dim=2)
        # Reshape to [1, 16, 768]
        return pooled.reshape(1, 16, 768).float()

    def extract_landmark_features(self, video_path: Path | str) -> Dict[str, torch.Tensor]:
        """Extract [1, 16, 225] MediaPipe landmarks and validity masks."""
        lm_out = self.landmark_extractor.extract(video_path)
        return {
            "landmarks": lm_out["landmarks"].unsqueeze(0).to(self.device),
            "pose_valid": lm_out["pose_valid"].unsqueeze(0).to(self.device),
            "left_valid": lm_out["left_valid"].unsqueeze(0).to(self.device),
            "right_valid": lm_out["right_valid"].unsqueeze(0).to(self.device),
        }

    @torch.inference_mode()
    def predict_video(
        self,
        video_path: Path | str,
        max_len: int = 30,
        block_ngram_repeat: int = 3,
    ) -> Dict[str, Any]:
        """Run full end-to-end translation pipeline on a video file.

        Returns dictionary with translation text, duration, feature shapes, and stage timings.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        t0 = time.perf_counter()

        # Stage 1: VideoMAE Feature Extraction
        t_vmae_start = time.perf_counter()
        rgb_features = self.extract_videomae_features(video_path)
        t_vmae_end = time.perf_counter()

        # Stage 2: MediaPipe Landmark Extraction
        t_lm_start = time.perf_counter()
        lm_dict = self.extract_landmark_features(video_path)
        t_lm_end = time.perf_counter()

        landmarks = lm_dict["landmarks"]
        pose_valid = lm_dict["pose_valid"]
        left_valid = lm_dict["left_valid"]
        right_valid = lm_dict["right_valid"]
        feature_mask = torch.ones((1, 16), dtype=torch.bool, device=self.device)

        # Stage 3: Autoregressive Decoding
        t_dec_start = time.perf_counter()
        text, token_ids = decode_autoregressive(
            model=self.model,
            rgb_features=rgb_features,
            landmarks=landmarks,
            pose_valid=pose_valid,
            left_valid=left_valid,
            right_valid=right_valid,
            feature_mask=feature_mask,
            vocab=self.vocab,
            max_len=max_len,
            block_ngram_repeat=block_ngram_repeat,
        )
        t_dec_end = time.perf_counter()

        total_ms = (t_dec_end - t0) * 1000.0
        vmae_ms = (t_vmae_end - t_vmae_start) * 1000.0
        lm_ms = (t_lm_end - t_lm_start) * 1000.0
        dec_ms = (t_dec_end - t_dec_start) * 1000.0

        return {
            "translation": text,
            "tokens": token_ids,
            "feature_shapes": {
                "rgb_features": list(rgb_features.shape),
                "landmarks": list(landmarks.shape),
                "pose_valid": list(pose_valid.shape),
            },
            "duration_ms": round(total_ms, 2),
            "stages_ms": {
                "videomae_ms": round(vmae_ms, 2),
                "landmarks_ms": round(lm_ms, 2),
                "decode_ms": round(dec_ms, 2),
            },
            "model_metadata": {
                "checkpoint": str(self.checkpoint_path.name),
                "epoch": self.checkpoint_epoch,
                "device": str(self.device),
                "vocab_size": len(self.vocab.itos),
            },
            "safeguards": {
                "max_len": max_len,
                "block_ngram_repeat": block_ngram_repeat,
                "note": "Safeguards constrain decoder repetition cycles; model weights are frozen.",
            },
        }

    def close(self) -> None:
        """Release MediaPipe and CUDA resources."""
        if hasattr(self, "landmark_extractor") and self.landmark_extractor:
            self.landmark_extractor.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
