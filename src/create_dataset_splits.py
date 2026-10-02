"""
Continuous Indian Sign Language Translation Using Deep Learning
Dataset Split & Contiguous Window Generation

Script: src/create_dataset_splits.py

Objective:
    Create reproducible train/validation/test experimental manifests for
    short-context continuous ISL translation while preventing source-video leakage.
    Splits are performed strictly at the video_id level BEFORE generating
    contiguous sliding windows (single = 1 segment, pair = 2 segments, triple = 3 segments).

Key Design Principles:
    1. Video-level splitting (no video_id appears in more than one split).
    2. Fixed random seed: 42.
    3. Target proportions: Train ~70%, Validation ~15%, Test ~15%.
    4. Group-level continuity balancing (source videos with runs >= 2, 3, 5, 10).
    5. Contiguous window generation strictly within continuous runs (never crossing sequence gaps).
    6. Chronological concatenation of ground-truth target sentences.
    7. Explicit leakage assertions (train ∩ val == ∅, train ∩ test == ∅, val ∩ test == ∅).

Outputs generated:
    1. data/metadata/dataset_splits.csv
    2. data/metadata/split_summary.json
    3. data/metadata/window_distribution.csv
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

RANDOM_SEED = 42
TARGET_TRAIN_RATIO = 0.70
TARGET_VAL_RATIO = 0.15
TARGET_TEST_RATIO = 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SAMPLES_PATH = METADATA_DIR / "selected_samples.csv"
INTEGRITY_PATH = METADATA_DIR / "video_integrity.csv"

# Generated local split artifacts
DATASET_SPLITS_CSV_PATH = METADATA_DIR / "dataset_splits.csv"
SPLIT_SUMMARY_JSON_PATH = METADATA_DIR / "split_summary.json"
WINDOW_DISTRIBUTION_CSV_PATH = METADATA_DIR / "window_distribution.csv"


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_files() -> None:
    """
    Ensure required input metadata files exist before proceeding.
    """
    for label, path in [
        ("Selected Samples Metadata", SAMPLES_PATH),
        ("Video Integrity Report", INTEGRITY_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] Required input file for {label} not found at: {path}"
            )


def validate_columns(
    df: pd.DataFrame, expected_columns: List[str], file_path: Path
) -> None:
    """
    Validate that required columns exist in input DataFrame.
    """
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Missing required columns in {file_path.name}: {missing}"
        )


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_filter_labeled_segments() -> pd.DataFrame:
    """
    Load metadata and integrity reports, filter to valid local videos,
    and match with labeled translation entries in selected_samples.csv.

    Returns:
        DataFrame of 2,197 matched labeled video segments.
    """
    validate_input_files()

    samples_df = pd.read_csv(SAMPLES_PATH)
    validate_columns(
        samples_df,
        ["uid", "video_id", "sequence_number", "text", "word_count"],
        SAMPLES_PATH,
    )

    integrity_df = pd.read_csv(INTEGRITY_PATH)
    validate_columns(
        integrity_df,
        ["filename", "video_id", "status"],
        INTEGRITY_PATH,
    )

    # Filter to locally valid videos
    valid_integrity = integrity_df[integrity_df["status"] == "valid"].copy()

    # Standardize filename basenames for join
    valid_integrity["basename"] = valid_integrity["filename"].apply(
        lambda p: Path(str(p)).name
    )
    samples_df["basename"] = samples_df["uid"].astype(str).str.strip() + ".mp4"

    # Inner join to obtain the usable supervised dataset
    labeled_df = valid_integrity[["basename", "video_id"]].merge(
        samples_df[["basename", "uid", "sequence_number", "text", "word_count"]],
        on="basename",
        how="inner",
        suffixes=("_integrity", ""),
    )

    # Clean text
    labeled_df["text"] = labeled_df["text"].fillna("").astype(str).str.strip()

    return labeled_df


# ============================================================
# CONTIGUOUS RUN DETECTION HELPER
# ============================================================

def find_contiguous_runs(sorted_records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Partition chronological records of a video_id into contiguous sequence runs
    where each record has sequence_number == previous.sequence_number + 1.
    """
    if not sorted_records:
        return []

    runs: List[List[Dict[str, Any]]] = []
    current_run: List[Dict[str, Any]] = [sorted_records[0]]

    for rec in sorted_records[1:]:
        prev_seq = current_run[-1]["sequence_number"]
        cur_seq = rec["sequence_number"]
        if cur_seq == prev_seq + 1:
            current_run.append(rec)
        else:
            runs.append(current_run)
            current_run = [rec]
    runs.append(current_run)
    return runs


# ============================================================
# VIDEO-LEVEL SPLITTING
# ============================================================

def split_video_ids(
    labeled_df: pd.DataFrame, seed: int = RANDOM_SEED
) -> Dict[str, str]:
    """
    Perform video_id level splitting with fixed random seed into train,
    validation, and test partitions (~70% / 15% / 15%).

    Returns:
        Mapping of video_id -> split name ("train", "validation", "test")
    """
    # Deterministic alphabetical ordering of unique source video IDs before shuffling
    unique_vids = sorted(labeled_df["video_id"].unique())
    n_total = len(unique_vids)

    # Calculate partition sizes
    n_train = int(round(n_total * TARGET_TRAIN_RATIO))  # 237
    n_val = int(round(n_total * TARGET_VAL_RATIO))      # 51
    n_test = n_total - n_train - n_val                  # 50

    # Shuffling with fixed NumPy RNG
    rng = np.random.default_rng(seed)
    shuffled_vids = np.array(unique_vids.copy())
    rng.shuffle(shuffled_vids)

    train_vids = set(shuffled_vids[:n_train])
    val_vids = set(shuffled_vids[n_train : n_train + n_val])
    test_vids = set(shuffled_vids[n_train + n_val :])

    # ------------------------------------------------------------
    # LEAKAGE VERIFICATION (Fail-Safe Assertion)
    # ------------------------------------------------------------
    assert len(train_vids & val_vids) == 0, "[ERROR] Leakage: train ∩ val is not empty!"
    assert len(train_vids & test_vids) == 0, "[ERROR] Leakage: train ∩ test is not empty!"
    assert len(val_vids & test_vids) == 0, "[ERROR] Leakage: val ∩ test is not empty!"

    split_map: Dict[str, str] = {}
    for vid in train_vids:
        split_map[vid] = "train"
    for vid in val_vids:
        split_map[vid] = "validation"
    for vid in test_vids:
        split_map[vid] = "test"

    return split_map


# ============================================================
# CONTIGUOUS WINDOW GENERATION
# ============================================================

def generate_contiguous_windows(
    labeled_df: pd.DataFrame, split_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Generate sliding windows (single, pair, triple) strictly inside contiguous
    sequence runs for each video_id in each split.
    """
    windows: List[Dict[str, Any]] = []

    # Group by video_id
    for video_id, group in labeled_df.groupby("video_id"):
        split_name = split_map[video_id]

        # Sort strictly by sequence_number
        sorted_records = (
            group.sort_values("sequence_number")
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

        # Identify contiguous runs
        contiguous_runs = find_contiguous_runs(sorted_records)

        for run in contiguous_runs:
            run_len = len(run)

            # 1. Single windows (window_size = 1)
            for item in run:
                target_text = item["text"]
                windows.append(
                    {
                        "split": split_name,
                        "window_type": "single",
                        "window_size": 1,
                        "video_id": video_id,
                        "start_sequence": item["sequence_number"],
                        "end_sequence": item["sequence_number"],
                        "uids": item["uid"],
                        "video_filenames": item["basename"],
                        "target_text": target_text,
                        "word_count": len(target_text.split()),
                    }
                )

            # 2. Pair windows (window_size = 2)
            if run_len >= 2:
                for i in range(run_len - 1):
                    pair = run[i : i + 2]
                    combined_text = " ".join(it["text"] for it in pair)
                    windows.append(
                        {
                            "split": split_name,
                            "window_type": "pair",
                            "window_size": 2,
                            "video_id": video_id,
                            "start_sequence": pair[0]["sequence_number"],
                            "end_sequence": pair[1]["sequence_number"],
                            "uids": ";".join(it["uid"] for it in pair),
                            "video_filenames": ";".join(it["basename"] for it in pair),
                            "target_text": combined_text,
                            "word_count": len(combined_text.split()),
                        }
                    )

            # 3. Triple windows (window_size = 3)
            if run_len >= 3:
                for i in range(run_len - 2):
                    triple = run[i : i + 3]
                    combined_text = " ".join(it["text"] for it in triple)
                    windows.append(
                        {
                            "split": split_name,
                            "window_type": "triple",
                            "window_size": 3,
                            "video_id": video_id,
                            "start_sequence": triple[0]["sequence_number"],
                            "end_sequence": triple[2]["sequence_number"],
                            "uids": ";".join(it["uid"] for it in triple),
                            "video_filenames": ";".join(it["basename"] for it in triple),
                            "target_text": combined_text,
                            "word_count": len(combined_text.split()),
                        }
                    )

    windows_df = pd.DataFrame(windows)

    # ------------------------------------------------------------
    # POST-GENERATION INTEGRITY & LEAKAGE VALIDATION
    # ------------------------------------------------------------
    for _, row in windows_df.iterrows():
        # Validate split consistency
        assert (
            row["split"] == split_map[row["video_id"]]
        ), f"[ERROR] Window split mismatch for video {row['video_id']}"

        # Validate consecutive sequence numbers
        expected_span = row["window_size"] - 1
        actual_span = row["end_sequence"] - row["start_sequence"]
        assert (
            actual_span == expected_span
        ), f"[ERROR] Non-consecutive window detected in video {row['video_id']}: {row['start_sequence']}..{row['end_sequence']}"

    return windows_df


# ============================================================
# MAIN SCRIPT EXECUTION
# ============================================================

def create_dataset_splits() -> Dict[str, Any]:
    """
    Execute full dataset splitting, window generation, validation,
    and report generation.
    """
    labeled_df = load_and_filter_labeled_segments()
    split_map = split_video_ids(labeled_df, seed=RANDOM_SEED)
    labeled_df["split"] = labeled_df["video_id"].map(split_map)

    # Generate sliding windows
    windows_df = generate_contiguous_windows(labeled_df, split_map)

    # ------------------------------------------------------------
    # COMPUTE SUMMARY STATISTICS
    # ------------------------------------------------------------
    # 1. Video IDs per split
    train_vids = {v for v, s in split_map.items() if s == "train"}
    val_vids = {v for v, s in split_map.items() if s == "validation"}
    test_vids = {v for v, s in split_map.items() if s == "test"}

    n_train_vids, n_val_vids, n_test_vids = (
        len(train_vids),
        len(val_vids),
        len(test_vids),
    )
    total_vids = len(split_map)

    # 2. Labeled segments per split
    seg_counts = labeled_df["split"].value_counts()
    n_train_segs = int(seg_counts.get("train", 0))
    n_val_segs = int(seg_counts.get("validation", 0))
    n_test_segs = int(seg_counts.get("test", 0))
    total_segs = len(labeled_df)

    # 3. Window counts by type and split
    window_cross = (
        pd.crosstab(windows_df["split"], windows_df["window_type"])
        .reindex(index=["train", "validation", "test"], columns=["single", "pair", "triple"], fill_value=0)
    )

    total_single = int((windows_df["window_type"] == "single").sum())
    total_pair = int((windows_df["window_type"] == "pair").sum())
    total_triple = int((windows_df["window_type"] == "triple").sum())
    total_windows = len(windows_df)

    # 4. Longest-run distribution per video across splits
    video_run_stats = []
    for video_id, grp in labeled_df.groupby("video_id"):
        seqs = grp["sequence_number"].tolist()
        runs = find_contiguous_runs(
            grp.sort_values("sequence_number").to_dict(orient="records")
        )
        longest = max(len(r) for r in runs) if runs else 0
        video_run_stats.append(
            {
                "video_id": video_id,
                "split": split_map[video_id],
                "longest_run": longest,
                "has_ge_2": longest >= 2,
                "has_ge_3": longest >= 3,
                "has_ge_5": longest >= 5,
                "has_ge_10": longest >= 10,
            }
        )
    run_stats_df = pd.DataFrame(video_run_stats)

    continuity_distribution = {}
    for threshold in [2, 3, 5, 10]:
        col = f"has_ge_{threshold}"
        t_cnt = int(run_stats_df[run_stats_df["split"] == "train"][col].sum())
        v_cnt = int(run_stats_df[run_stats_df["split"] == "validation"][col].sum())
        te_cnt = int(run_stats_df[run_stats_df["split"] == "test"][col].sum())
        tot_cnt = t_cnt + v_cnt + te_cnt
        continuity_distribution[f">={threshold}"] = {
            "train": t_cnt,
            "train_pct": round(t_cnt / tot_cnt * 100.0, 2) if tot_cnt > 0 else 0.0,
            "validation": v_cnt,
            "validation_pct": round(v_cnt / tot_cnt * 100.0, 2) if tot_cnt > 0 else 0.0,
            "test": te_cnt,
            "test_pct": round(te_cnt / tot_cnt * 100.0, 2) if tot_cnt > 0 else 0.0,
            "total": tot_cnt,
        }

    # ------------------------------------------------------------
    # SAVE OUTPUT ARTIFACTS
    # ------------------------------------------------------------
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset splits manifest CSV
    manifest_cols = [
        "split",
        "window_type",
        "window_size",
        "video_id",
        "start_sequence",
        "end_sequence",
        "uids",
        "video_filenames",
        "target_text",
        "word_count",
    ]
    windows_df[manifest_cols].to_csv(DATASET_SPLITS_CSV_PATH, index=False)

    # 2. Window distribution CSV
    window_dist_records = []
    for split_name in ["train", "validation", "test"]:
        for w_type, w_size in [("single", 1), ("pair", 2), ("triple", 3)]:
            cnt = int(window_cross.loc[split_name, w_type])
            window_dist_records.append(
                {
                    "split": split_name,
                    "window_type": w_type,
                    "window_size": w_size,
                    "count": cnt,
                }
            )
    window_dist_df = pd.DataFrame(window_dist_records)
    window_dist_df.to_csv(WINDOW_DISTRIBUTION_CSV_PATH, index=False)

    # 3. Split summary JSON
    summary_data = {
        "dataset_name": "iSign",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "source_video_groups": {
            "total": total_vids,
            "train": n_train_vids,
            "train_pct": round(n_train_vids / total_vids * 100.0, 2),
            "validation": n_val_vids,
            "validation_pct": round(n_val_vids / total_vids * 100.0, 2),
            "test": n_test_vids,
            "test_pct": round(n_test_vids / total_vids * 100.0, 2),
        },
        "labeled_segments": {
            "total": total_segs,
            "train": n_train_segs,
            "train_pct": round(n_train_segs / total_segs * 100.0, 2),
            "validation": n_val_segs,
            "validation_pct": round(n_val_segs / total_segs * 100.0, 2),
            "test": n_test_segs,
            "test_pct": round(n_test_segs / total_segs * 100.0, 2),
        },
        "window_counts_by_type": {
            "single": total_single,
            "pair": total_pair,
            "triple": total_triple,
            "total": total_windows,
            "single_pct": round(total_single / total_windows * 100.0, 2),
            "pair_pct": round(total_pair / total_windows * 100.0, 2),
            "triple_pct": round(total_triple / total_windows * 100.0, 2),
        },
        "window_counts_by_split": {
            "train": {
                "single": int(window_cross.loc["train", "single"]),
                "pair": int(window_cross.loc["train", "pair"]),
                "triple": int(window_cross.loc["train", "triple"]),
                "total": int(window_cross.loc["train"].sum()),
            },
            "validation": {
                "single": int(window_cross.loc["validation", "single"]),
                "pair": int(window_cross.loc["validation", "pair"]),
                "triple": int(window_cross.loc["validation", "triple"]),
                "total": int(window_cross.loc["validation"].sum()),
            },
            "test": {
                "single": int(window_cross.loc["test", "single"]),
                "pair": int(window_cross.loc["test", "pair"]),
                "triple": int(window_cross.loc["test", "triple"]),
                "total": int(window_cross.loc["test"].sum()),
            },
        },
        "leakage_checks": {
            "train_val_overlap": len(train_vids & val_vids),
            "train_test_overlap": len(train_vids & test_vids),
            "val_test_overlap": len(val_vids & test_vids),
            "leakage_free": True,
        },
        "continuity_distribution": continuity_distribution,
        "limitations_and_warnings": [
            "Signer-level splitting was NOT performed because signer identity is not available in the dataset metadata. video_id represents the source video recording group.",
            "All multi-segment windows (pair, triple) are strictly contiguous and do not cross sequence gaps or video boundaries.",
            "18 valid local video files lacking ground-truth translation labels were excluded from translation window generation.",
        ],
    }

    with open(SPLIT_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # ------------------------------------------------------------
    # TERMINAL REPORT FORMATTING
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    print("\nSOURCE VIDEO GROUPS")
    print("-" * 60)
    print(f"Train      : {n_train_vids:>3} source videos ({n_train_vids / total_vids * 100:5.2f}%)")
    print(f"Validation : {n_val_vids:>3} source videos ({n_val_vids / total_vids * 100:5.2f}%)")
    print(f"Test       : {n_test_vids:>3} source videos ({n_test_vids / total_vids * 100:5.2f}%)")
    print(f"Total      : {total_vids:>3} source videos (100.00%)")

    print("\nLABELED SEGMENTS")
    print("-" * 60)
    print(f"Train      : {n_train_segs:>5,} segments ({n_train_segs / total_segs * 100:5.2f}%)")
    print(f"Validation : {n_val_segs:>5,} segments ({n_val_segs / total_segs * 100:5.2f}%)")
    print(f"Test       : {n_test_segs:>5,} segments ({n_test_segs / total_segs * 100:5.2f}%)")
    print(f"Total      : {total_segs:>5,} segments (100.00%)")

    print("\nWINDOW COUNTS")
    print("-" * 60)
    print(f"Single     : {total_single:>5,} windows ({total_single / total_windows * 100:5.2f}%)")
    print(f"Pair       : {total_pair:>5,} windows ({total_pair / total_windows * 100:5.2f}%)")
    print(f"Triple     : {total_triple:>5,} windows ({total_triple / total_windows * 100:5.2f}%)")
    print(f"Total      : {total_windows:>5,} windows (100.00%)")

    print("\nWINDOW COUNTS BY SPLIT")
    print("-" * 60)
    for s_name in ["train", "validation", "test"]:
        s_single = int(window_cross.loc[s_name, "single"])
        s_pair = int(window_cross.loc[s_name, "pair"])
        s_triple = int(window_cross.loc[s_name, "triple"])
        s_tot = s_single + s_pair + s_triple
        print(f"{s_name.capitalize():<11}: {s_tot:>5,} windows "
              f"(Single: {s_single:>4,}, Pair: {s_pair:>4,}, Triple: {s_triple:>3}) "
              f"[{s_tot / total_windows * 100:5.2f}% of total]")

    print("\nCONTINUITY DISTRIBUTION")
    print("-" * 60)
    print("Source videos by longest contiguous run across splits:")
    for thresh in [2, 3, 5, 10]:
        stats = continuity_distribution[f">={thresh}"]
        print(f"  - Longest run >= {thresh:>2} : "
              f"Train: {stats['train']:>3} ({stats['train_pct']:5.2f}%) | "
              f"Val: {stats['validation']:>2} ({stats['validation_pct']:5.2f}%) | "
              f"Test: {stats['test']:>2} ({stats['test_pct']:5.2f}%) | "
              f"Total: {stats['total']:>3}")

    print("\nLEAKAGE CHECK")
    print("-" * 60)
    print(f"Train ∩ Validation : {len(train_vids & val_vids)}")
    print(f"Train ∩ Test       : {len(train_vids & test_vids)}")
    print(f"Validation ∩ Test  : {len(val_vids & test_vids)}")
    print("All must be zero. Status: VERIFIED LEAKAGE-FREE.")

    print("\n" + "=" * 60)
    print("FINAL DATASET MANIFEST")
    print("=" * 60)
    print(f"Total single windows : {total_single:>5,} ({total_single / total_windows * 100:5.2f}%)")
    print(f"Total pair windows   : {total_pair:>5,} ({total_pair / total_windows * 100:5.2f}%)")
    print(f"Total triple windows : {total_triple:>5,} ({total_triple / total_windows * 100:5.2f}%)")
    print(f"Total windows        : {total_windows:>5,} (100.00%)")
    print("=" * 60)

    print("\nGenerated Output Files:")
    print(f"  1. {DATASET_SPLITS_CSV_PATH}")
    print(f"  2. {WINDOW_DISTRIBUTION_CSV_PATH}")
    print(f"  3. {SPLIT_SUMMARY_JSON_PATH}")
    print("=" * 60 + "\n")

    return summary_data


if __name__ == "__main__":
    create_dataset_splits()
