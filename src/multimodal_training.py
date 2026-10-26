
"""Batching and loss helpers for multimodal ISL training."""

import torch
from torch.nn.utils.rnn import pad_sequence


def make_multimodal_collate_fn(vocab, max_target_len=None):
    """Pad RGB, landmarks, masks, and target text consistently within each batch."""
    pad_id = vocab.pad_id
    eos_id = vocab.eos_id

    def collate(batch):
        rgb_parts = [item["rgb_features"] for item in batch]
        landmark_parts = [item["landmarks"] for item in batch]

        rgb_lengths = torch.tensor(
            [x.shape[0] for x in rgb_parts], dtype=torch.long
        )
        landmark_lengths = torch.tensor(
            [x.shape[0] for x in landmark_parts], dtype=torch.long
        )

        if not torch.equal(rgb_lengths, landmark_lengths):
            raise ValueError("RGB and landmark sequence lengths do not match.")

        rgb = pad_sequence(rgb_parts, batch_first=True, padding_value=0.0)
        landmarks = pad_sequence(
            landmark_parts, batch_first=True, padding_value=0.0
        )

        lengths = rgb_lengths
        feature_mask = (
            torch.arange(rgb.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        )

        def pad_mask(key):
            masks = [item[key].bool() for item in batch]
            return pad_sequence(masks, batch_first=True, padding_value=False)

        encoded = []
        for item in batch:
            ids = vocab.encode(item["target_text"])
            if max_target_len is not None and len(ids) > max_target_len:
                ids = ids[:max_target_len - 1] + [eos_id]
            encoded.append(torch.tensor(ids, dtype=torch.long))

        target_ids = pad_sequence(
            encoded, batch_first=True, padding_value=pad_id
        )

        return {
            "rgb_features": rgb,
            "landmarks": landmarks,
            "pose_valid": pad_mask("pose_valid"),
            "left_valid": pad_mask("left_valid"),
            "right_valid": pad_mask("right_valid"),
            "feature_mask": feature_mask,
            "target_ids": target_ids,
            "target_text": [item["target_text"] for item in batch],
            "video_id": [item["video_id"] for item in batch],
            "window_type": [item["window_type"] for item in batch],
            "split": [item["split"] for item in batch],
        }

    return collate


def multimodal_sequence_cross_entropy(logits, target_ids, pad_id):
    """Cross-entropy over target tokens, ignoring padding."""
    labels = target_ids[:, 1:]

    if logits.numel() == 0 or labels.numel() == 0:
        return logits.sum() * 0.0

    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def token_accuracy_counts(logits, target_ids, pad_id):
    """Teacher-forced token counts, ignoring pad labels.

    This is token-level teacher-forcing accuracy, not sentence BLEU/WER.
    """
    labels = target_ids[:, 1:]
    if logits.numel() == 0 or labels.numel() == 0:
        return 0, 0
    valid = labels.ne(pad_id)
    correct = (logits.argmax(dim=-1).eq(labels) & valid).sum().item()
    total = int(valid.sum().item())
    return int(correct), total


def token_accuracy(logits, target_ids, pad_id):
    """Mean teacher-forced token accuracy over non-pad labels."""
    correct, total = token_accuracy_counts(logits, target_ids, pad_id)
    if total == 0:
        return 0.0
    return correct / total
