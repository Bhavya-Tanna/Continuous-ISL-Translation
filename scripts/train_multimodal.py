
"""Resumable RGB + MediaPipe multimodal ISL translation training."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multimodal_dataset import ISLMultimodalDataset
from src.multimodal_model import ISLMultimodalModel
from src.multimodal_training import (
    make_multimodal_collate_fn,
    multimodal_sequence_cross_entropy,
    token_accuracy_counts,
)
from src.text import build_vocab, Vocabulary
from src.training import save_json, corpus_bleu, word_error_rate


def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Train the RGB + MediaPipe ISL translation model."
    )
    p.add_argument("--manifest", default="data/metadata/dataset_splits.csv")
    p.add_argument("--video-dir", default="data/raw/videos")
    p.add_argument(
        "--rgb-feature-dir",
        default="data/processed/videomae_features_16f",
        help="Cached VideoMAE-Base features [8,768] per video.",
    )
    p.add_argument(
        "--landmark-dir",
        default="data/processed/mediapipe_landmarks",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed/multimodal_run_v4",
        help="New experiment directory. Does not overwrite previous runs.",
    )
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial AdamW learning rate (current project value).",
    )
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--word-dropout", type=float, default=0.20, help="Decoder input token-dropout probability during training.")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--patience", type=int, default=9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-decode-len", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="Smoke testing only; 0 uses the full training dataset.",
    )
    p.add_argument(
        "--limit-val-batches",
        type=int,
        default=0,
        help="Smoke testing only; 0 uses the full validation dataset.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(args)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(path, payload):
    """Save a checkpoint through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    pad_id,
    grad_clip,
    limit_batches=0,
):
    model.train()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        rgb = batch["rgb_features"].to(device)
        landmarks = batch["landmarks"].to(device)
        target_ids = batch["target_ids"].to(device)
        feature_mask = batch["feature_mask"].to(device)
        pose_valid = batch["pose_valid"].to(device)
        left_valid = batch["left_valid"].to(device)
        right_valid = batch["right_valid"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            rgb,
            landmarks,
            target_ids,
            feature_mask=feature_mask,
            pose_valid=pose_valid,
            left_valid=left_valid,
            right_valid=right_valid,
        )

        loss = multimodal_sequence_cross_entropy(
            logits, target_ids, pad_id
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss: {loss.item()}"
            )

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )

        optimizer.step()
        total_loss += loss.item()
        correct, total = token_accuracy_counts(logits.detach(), target_ids, pad_id)
        correct_tokens += correct
        total_tokens += total
        batches += 1

        if limit_batches and batches >= limit_batches:
            break

    if batches == 0:
        raise RuntimeError("No training batches were processed.")

    token_acc = correct_tokens / total_tokens if total_tokens else 0.0
    return total_loss / batches, token_acc


@torch.inference_mode()
def evaluate_multimodal(
    model, loader, vocab, device, max_decode_len, limit_batches=0
):
    model.eval()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    batches = 0
    references = []
    hypotheses = []
    examples = []

    for batch in tqdm(loader, desc="validation", leave=False):
        rgb = batch["rgb_features"].to(device)
        landmarks = batch["landmarks"].to(device)
        target_ids = batch["target_ids"].to(device)
        feature_mask = batch["feature_mask"].to(device)
        pose_valid = batch["pose_valid"].to(device)
        left_valid = batch["left_valid"].to(device)
        right_valid = batch["right_valid"].to(device)

        logits = model(
            rgb,
            landmarks,
            target_ids,
            feature_mask=feature_mask,
            pose_valid=pose_valid,
            left_valid=left_valid,
            right_valid=right_valid,
        )

        loss = multimodal_sequence_cross_entropy(
            logits, target_ids, vocab.pad_id
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss: {loss.item()}"
            )

        total_loss += loss.item()
        correct, total = token_accuracy_counts(logits, target_ids, vocab.pad_id)
        correct_tokens += correct
        total_tokens += total
        batches += 1

        predicted_ids = model.greedy_decode(
            rgb,
            landmarks,
            feature_mask=feature_mask,
            pose_valid=pose_valid,
            left_valid=left_valid,
            right_valid=right_valid,
            max_len=max_decode_len,
        )

        predictions = [
            vocab.decode(row.tolist()) for row in predicted_ids
        ]
        references.extend(batch["target_text"])
        hypotheses.extend(predictions)

        for ref, hyp, vid, window_type in zip(
            batch["target_text"],
            predictions,
            batch["video_id"],
            batch["window_type"],
        ):
            if len(examples) < 100:
                examples.append({
                    "video_id": vid,
                    "window_type": window_type,
                    "reference": ref,
                    "prediction": hyp,
                })

        if limit_batches and batches >= limit_batches:
            break

    token_acc = correct_tokens / total_tokens if total_tokens else 0.0
    metrics = {
        "loss": total_loss / max(batches, 1),
        "token_accuracy": token_acc,
        "bleu": corpus_bleu(references, hypotheses) if references else 0.0,
        "wer": word_error_rate(references, hypotheses) if references else 0.0,
        "n_examples": len(references),
        "n_tokens": total_tokens,
    }

    return metrics, examples


def main():
    args = parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if args.patience < 1 or args.lr_patience < 0:
        raise ValueError("Invalid early-stopping or LR patience.")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    evaluation_dir = output_dir / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    vocab_path = output_dir / "vocab.json"
    config_path = output_dir / "config.json"
    history_path = output_dir / "history.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                output_dir / "train.log", encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    train_ds = ISLMultimodalDataset(
        args.manifest,
        "train",
        args.video_dir,
        args.rgb_feature_dir,
        args.landmark_dir,
    )
    val_ds = ISLMultimodalDataset(
        args.manifest,
        "validation",
        args.video_dir,
        args.rgb_feature_dir,
        args.landmark_dir,
    )

    if args.resume:
        if not last_path.is_file() or not vocab_path.is_file():
            raise FileNotFoundError(
                "Resume requires checkpoints/last.pt and vocab.json."
            )
        blob = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        config = blob["config"]
        vocab = Vocabulary.load(vocab_path)
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file()
            else []
        )
        start_epoch = int(blob["epoch"]) + 1
        best_val = float(
            blob.get("best_val_loss", blob["val_loss"])
        )
        bad_epochs = int(
            blob.get("epochs_without_improvement", 0)
        )
    else:
        if last_path.exists():
            raise FileExistsError(
                f"{last_path} already exists. Use --resume or a new "
                "--output-dir to avoid overwriting a previous run."
            )

        vocab = build_vocab(
            (row["target_text"] for row in train_ds.rows),
            min_freq=1,
        )
        vocab.save(vocab_path)

        config = {
            "model": "ISLMultimodalModel",
            "rgb_dim": 768,
            "landmark_dim": 225,
            "fusion_dim": 512,
            "encoder_hidden": 256,
            "decoder_hidden": 512,
            "embedding_dim": 256,
            "attention_dim": 256,
            "dropout": args.dropout,
            "word_dropout": args.word_dropout,
            "use_embedding_in_output": False,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "early_stopping_patience": args.patience,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "epochs": args.epochs,
            "manifest": args.manifest,
            "video_dir": args.video_dir,
            "rgb_feature_dir": args.rgb_feature_dir,
            "landmark_dir": args.landmark_dir,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_mode": "min",
        }
        history = []
        start_epoch = 1
        best_val = float("inf")
        bad_epochs = 0

    # Use the saved settings when resuming, so the run stays consistent.
    collate_fn = make_multimodal_collate_fn(vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

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
        dropout=config["dropout"],
        use_embedding_in_output=config["use_embedding_in_output"],
        word_dropout=config.get("word_dropout", args.word_dropout),
        unk_id=vocab.unk_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("lr_factor", args.lr_factor),
        patience=config.get("lr_patience", args.lr_patience),
        min_lr=config.get("min_lr", args.min_lr),
    )

    if args.resume:
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        if blob.get("scheduler") is not None:
            scheduler.load_state_dict(blob["scheduler"])

    config["vocab_size"] = len(vocab.itos)
    save_json(config_path, config)

    logging.info(
        "Train windows=%d | validation windows=%d | vocab=%d",
        len(train_ds),
        len(val_ds),
        len(vocab.itos),
    )
    logging.info(
        "Active config: lr=%.2e min_lr=%.2e factor=%.2f "
        "lr_patience=%d early_stop=%d word_dropout=%.2f batch_size=%d max_epochs=%d "
        "weight_decay=%.2e rgb_feature_dir=%s",
        config["lr"],
        config.get("min_lr", args.min_lr),
        config.get("lr_factor", args.lr_factor),
        config.get("lr_patience", args.lr_patience),
        config.get("early_stopping_patience", args.patience),
        config.get("word_dropout", args.word_dropout),
        args.batch_size,
        args.epochs,
        config.get("weight_decay", args.weight_decay),
        config.get("rgb_feature_dir", args.rgb_feature_dir),
    )
    if "resnet" in str(config.get("rgb_feature_dir", args.rgb_feature_dir)).lower():
        logging.warning(
            "rgb_feature_dir looks like a ResNet cache, but the model "
            "expects VideoMAE [8,768] features."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_token_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            vocab.pad_id,
            config.get("grad_clip", args.grad_clip),
            args.limit_train_batches,
        )

        val_metrics, val_examples = evaluate_multimodal(
            model,
            val_loader,
            vocab,
            device,
            args.max_decode_len,
            args.limit_val_batches,
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        gpu_mem = (
            round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            if device.type == "cuda"
            else 0.0
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_accuracy": train_token_acc,
            "val_loss": val_metrics["loss"],
            "val_token_accuracy": val_metrics["token_accuracy"],
            "val_bleu": val_metrics["bleu"],
            "val_wer": val_metrics["wer"],
            "learning_rate": current_lr,
            "epochs_without_improvement": bad_epochs,
            "seconds": round(time.time() - epoch_start, 2),
            "gpu_memory_mb": gpu_mem,
        }
        history.append(row)

        save_json(history_path, history)
        save_json(
            evaluation_dir / "val_examples.json", val_examples
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "best_val_loss": best_val,
            "epochs_without_improvement": bad_epochs,
            "config": config,
        }

        # Always save the latest recoverable state.
        atomic_torch_save(last_path, checkpoint)

        # Keep best.pt unchanged unless validation loss improves.
        if improved:
            atomic_torch_save(best_path, checkpoint)

        logging.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f "
            "train_token_acc=%.4f val_token_acc=%.4f "
            "(token-level teacher-forcing accuracy, not sentence accuracy) "
            "val_BLEU=%.4f val_WER=%.4f lr=%.2e bad_epochs=%d time=%.1fs",
            epoch,
            train_loss,
            val_metrics["loss"],
            train_token_acc,
            val_metrics["token_accuracy"],
            val_metrics["bleu"],
            val_metrics["wer"],
            current_lr,
            bad_epochs,
            row["seconds"],
        )

        if bad_epochs >= config.get(
            "early_stopping_patience", args.patience
        ):
            logging.info(
                "Early stopping at epoch %d: validation loss did not "
                "improve for %d consecutive epochs.",
                epoch,
                bad_epochs,
            )
            break

    logging.info(
        "Training finished. Best validation loss: %.4f", best_val
    )
    logging.info("Best checkpoint: %s", best_path)
    logging.info("Latest checkpoint: %s", last_path)


if __name__ == "__main__":
    main()
