
"""RGB + MediaPipe fusion model for continuous ISL translation."""

import torch
from torch import nn

from .baseline_model import ISLSeq2SeqBaseline


class ISLMultimodalModel(nn.Module):
    """Fuse video features and pose/hand landmarks before translation."""

    def __init__(
        self,
        vocab_size,
        rgb_dim=512,
        landmark_dim=225,
        fusion_dim=512,
        encoder_hidden=256,
        decoder_hidden=512,
        embedding_dim=256,
        attention_dim=256,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        dropout=0.1,
        use_embedding_in_output=False,
        word_dropout=0.0,
        unk_id=3,
    ):
        super().__init__()

        if rgb_dim not in (512, 768):
            raise ValueError("rgb_dim must be 512 (ResNet) or 768 (VideoMAE).")
        if landmark_dim != 225:
            raise ValueError("Expected landmark dimension 225.")

        self.rgb_dim = rgb_dim
        self.landmark_dim = landmark_dim
        self.fusion_dim = fusion_dim

        # Landmark groups: pose (99), left hand (63), right hand (63).
        self.landmark_projection = nn.Sequential(
            nn.Linear(landmark_dim, fusion_dim, bias=False),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mask_projection = nn.Linear(3, fusion_dim, bias=False)

        self.gate = nn.Linear(
            rgb_dim + fusion_dim + 3,
            fusion_dim,
        )

        self.rgb_projection = (
            nn.Identity()
            if rgb_dim == fusion_dim
            else nn.Linear(rgb_dim, fusion_dim)
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        self.decoder = ISLSeq2SeqBaseline(
            vocab_size=vocab_size,
            input_dim=fusion_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            dropout=dropout,
            use_embedding_in_output=use_embedding_in_output,
            word_dropout=word_dropout,
            unk_id=unk_id,
        )

    def fuse(
        self,
        rgb_features,
        landmarks,
        pose_valid,
        left_valid,
        right_valid,
    ):
        """Fuse aligned [B,T,rgb_dim] and [B,T,225] sequences."""
        if (
            rgb_features.ndim != 3
            or rgb_features.shape[-1] != self.rgb_dim
        ):
            raise ValueError(
                f"rgb_features must have shape [B,T,{self.rgb_dim}]."
            )

        if (
            landmarks.ndim != 3
            or landmarks.shape[-1] != self.landmark_dim
        ):
            raise ValueError("landmarks must have shape [B,T,225].")

        if rgb_features.shape[:2] != landmarks.shape[:2]:
            raise ValueError(
                "RGB and landmark batch/temporal dimensions must match."
            )

        expected_mask_shape = rgb_features.shape[:2]
        masks = []

        for name, mask in (
            ("pose_valid", pose_valid),
            ("left_valid", left_valid),
            ("right_valid", right_valid),
        ):
            if mask.shape != expected_mask_shape:
                raise ValueError(f"{name} must have shape [B,T].")

            masks.append(
                mask.to(
                    device=landmarks.device,
                    dtype=landmarks.dtype,
                )
            )

        pose_mask, left_mask, right_mask = masks

        # Mask unavailable pose/hand coordinates.
        pose = landmarks[..., :99] * pose_mask.unsqueeze(-1)
        left = landmarks[..., 99:162] * left_mask.unsqueeze(-1)
        right = landmarks[..., 162:225] * right_mask.unsqueeze(-1)
        masked_landmarks = torch.cat((pose, left, right), dim=-1)

        mask_features = torch.stack(masks, dim=-1)

        projected_landmarks = self.landmark_projection(masked_landmarks)
        projected_landmarks = (
            projected_landmarks + self.mask_projection(mask_features)
        )

        rgb = self.rgb_projection(rgb_features)

        gate_input = torch.cat(
            (rgb_features, projected_landmarks, mask_features),
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))

        fused = self.fusion_norm(rgb + gate * projected_landmarks)
        return self.dropout(fused)

    def forward(
        self,
        rgb_features,
        landmarks,
        target_ids,
        feature_mask=None,
        pose_valid=None,
        left_valid=None,
        right_valid=None,
    ):
        batch_frames = rgb_features.shape[:2]

        if pose_valid is None:
            pose_valid = torch.ones(
                batch_frames,
                dtype=torch.bool,
                device=rgb_features.device,
            )
        if left_valid is None:
            left_valid = torch.ones_like(pose_valid)
        if right_valid is None:
            right_valid = torch.ones_like(pose_valid)

        fused = self.fuse(
            rgb_features,
            landmarks,
            pose_valid,
            left_valid,
            right_valid,
        )

        return self.decoder(
            fused,
            target_ids,
            feature_mask=feature_mask,
        )

    @torch.inference_mode()
    def greedy_decode(
        self,
        rgb_features,
        landmarks,
        feature_mask=None,
        pose_valid=None,
        left_valid=None,
        right_valid=None,
        max_len=60,
    ):
        was_training = self.training
        self.eval()

        try:
            batch_frames = rgb_features.shape[:2]

            if pose_valid is None:
                pose_valid = torch.ones(
                    batch_frames,
                    dtype=torch.bool,
                    device=rgb_features.device,
                )
            if left_valid is None:
                left_valid = torch.ones_like(pose_valid)
            if right_valid is None:
                right_valid = torch.ones_like(pose_valid)

            fused = self.fuse(
                rgb_features,
                landmarks,
                pose_valid,
                left_valid,
                right_valid,
            )

            return self.decoder.greedy_decode(
                fused,
                feature_mask=feature_mask,
                max_len=max_len,
            )

        finally:
            self.train(was_training)
