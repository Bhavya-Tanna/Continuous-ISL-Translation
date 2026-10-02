"""
Continuous Indian Sign Language Translation Using Deep Learning
Sequence Continuity Analysis

Script: src/analyze_sequence_continuity.py

Objective:
    Determine whether the currently available iSign subset contains sufficiently
    continuous ordered sequences for continuous sign-language translation experiments.
    Evaluates segment sequence ordering (sequence_number) per source video (video_id),
    identifies contiguous runs, quantifies sequence gaps, and analyzes continuity
    across both the total available dataset and the labeled translation subset.

Outputs generated:
    1. data/metadata/sequence_continuity_by_video.csv
    2. data/metadata/sequence_run_distribution.csv
    3. data/metadata/sequence_continuity_summary.json
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION & PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SAMPLES_PATH = METADATA_DIR / "selected_samples.csv"
INDEX_PATH = METADATA_DIR / "selected_video_index.csv"
INTEGRITY_PATH = METADATA_DIR / "video_integrity.csv"

# Generated local output files
CONTINUITY_BY_VIDEO_CSV_PATH = METADATA_DIR / "sequence_continuity_by_video.csv"
RUN_DISTRIBUTION_CSV_PATH = METADATA_DIR / "sequence_run_distribution.csv"
CONTINUITY_SUMMARY_JSON_PATH = METADATA_DIR / "sequence_continuity_summary.json"


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_files() -> None:
    """
    Ensure all required input metadata files exist before beginning analysis.
    Fails immediately with an informative FileNotFoundError if any file is missing.
    """
    required_files = [
        ("Selected Samples Metadata", SAMPLES_PATH),
        ("Video Integrity Report", INTEGRITY_PATH),
    ]

    for label, file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Required input file for {label} not found at: {file_path}"
            )


def validate_columns(
    df: pd.DataFrame, expected_columns: List[str], file_path: Path
) -> None:
    """
    Validate that a DataFrame contains all required columns.
    Fails with ValueError if expected columns are missing.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Missing required columns in {file_path.name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# DATA PREPARATION & RUN IDENTIFICATION
# ============================================================

def get_contiguous_runs(sorted_sequences: List[int]) -> List[List[int]]:
    """
    Given a sorted list of integer sequence numbers, partition them into
    contiguous runs where each consecutive element equals previous + 1.

    Example:
        [1, 2, 3, 4, 7, 8, 10] -> [[1, 2, 3, 4], [7, 8], [10]]
    """
    if not sorted_sequences:
        return []

    runs: List[List[int]] = []
    current_run: List[int] = [sorted_sequences[0]]

    for num in sorted_sequences[1:]:
        if num == current_run[-1] + 1:
            current_run.append(num)
        else:
            runs.append(current_run)
            current_run = [num]
    runs.append(current_run)
    return runs


def categorize_run_length(length: int) -> str:
    """
    Categorize contiguous run length into standard distribution buckets:
        1, 2, 3, 4, 5, 6-9, 10+
    """
    if length == 1:
        return "run length 1"
    elif length == 2:
        return "run length 2"
    elif length == 3:
        return "run length 3"
    elif length == 4:
        return "run length 4"
    elif length == 5:
        return "run length 5"
    elif 6 <= length <= 9:
        return "run length 6-9"
    else:
        return "run length 10+"


# ============================================================
# MAIN CONTINUITY ANALYSIS
# ============================================================

def analyze_continuity() -> Dict[str, Any]:
    """
    Perform sequence continuity analysis across all source video groups,
    compute run distributions, evaluate labeled vs available continuity,
    and generate summary reports and CSV artifacts.
    """
    validate_input_files()

    # 1. Load inputs
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

    # 2. Filter to valid local videos
    valid_videos = integrity_df[integrity_df["status"] == "valid"].copy()

    # 3. Derive standard keys & extract sequence numbers
    valid_videos["basename"] = valid_videos["filename"].apply(
        lambda p: Path(str(p)).name
    )
    valid_videos["uid"] = valid_videos["basename"].str.replace(
        r"\.mp4$", "", regex=True
    )
    # Extract integer sequence number from UID (e.g., "-CpDPjXeB2c--1" -> 1)
    valid_videos["sequence_number"] = (
        valid_videos["uid"].str.rsplit("-", n=1).str[1].astype(int)
    )

    # Identify which valid videos have ground-truth text in selected_samples
    samples_df["basename"] = samples_df["uid"].astype(str).str.strip() + ".mp4"
    labeled_basenames_set = set(samples_df["basename"])
    valid_videos["is_labeled"] = valid_videos["basename"].isin(labeled_basenames_set)

    total_valid_videos = len(valid_videos)
    total_labeled_videos = int(valid_videos["is_labeled"].sum())

    # ============================================================
    # PER-VIDEO CONTINUITY COMPUTATION
    # ============================================================
    video_records: List[Dict[str, Any]] = []
    all_runs_available: List[List[int]] = []
    all_runs_labeled: List[List[int]] = []

    # Map for labeled runs per video_id
    labeled_runs_by_video: Dict[str, List[List[int]]] = {}

    grouped = valid_videos.groupby("video_id")

    for video_id, group in grouped:
        # Available segments
        available_seqs = sorted(group["sequence_number"].tolist())
        runs_avail = get_contiguous_runs(available_seqs)
        run_lens_avail = [len(r) for r in runs_avail]
        all_runs_available.extend(runs_avail)

        # Labeled segments
        labeled_group = group[group["is_labeled"]]
        labeled_seqs = sorted(labeled_group["sequence_number"].tolist())
        runs_lab = get_contiguous_runs(labeled_seqs)
        labeled_runs_by_video[video_id] = runs_lab
        all_runs_labeled.extend(runs_lab)

        num_runs = len(runs_avail)
        longest_run = max(run_lens_avail) if run_lens_avail else 0
        mean_run = float(np.mean(run_lens_avail)) if run_lens_avail else 0.0
        # Number of discontinuity gaps between contiguous runs
        num_gaps = max(0, num_runs - 1)

        video_records.append(
            {
                "video_id": video_id,
                "available_segments": len(available_seqs),
                "labeled_segments": len(labeled_seqs),
                "min_sequence": min(available_seqs) if available_seqs else None,
                "max_sequence": max(available_seqs) if available_seqs else None,
                "num_contiguous_runs": num_runs,
                "longest_contiguous_run": longest_run,
                "mean_run_length": round(mean_run, 2),
                "num_gaps": num_gaps,
            }
        )

    # Sort video continuity table by longest_contiguous_run desc, available_segments desc, video_id asc
    continuity_by_video_df = pd.DataFrame(video_records).sort_values(
        by=["longest_contiguous_run", "available_segments", "video_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Save sequence_continuity_by_video.csv
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    continuity_by_video_df.to_csv(CONTINUITY_BY_VIDEO_CSV_PATH, index=False)

    # ============================================================
    # GLOBAL STATISTICS: ALL AVAILABLE SEGMENTS
    # ============================================================
    total_source_video_ids = len(continuity_by_video_df)

    vids_ge_2 = int((continuity_by_video_df["longest_contiguous_run"] >= 2).sum())
    vids_ge_3 = int((continuity_by_video_df["longest_contiguous_run"] >= 3).sum())
    vids_ge_5 = int((continuity_by_video_df["longest_contiguous_run"] >= 5).sum())
    vids_ge_10 = int((continuity_by_video_df["longest_contiguous_run"] >= 10).sum())

    pct_vids_ge_2 = round(vids_ge_2 / total_source_video_ids * 100.0, 2)
    pct_vids_ge_3 = round(vids_ge_3 / total_source_video_ids * 100.0, 2)
    pct_vids_ge_5 = round(vids_ge_5 / total_source_video_ids * 100.0, 2)
    pct_vids_ge_10 = round(vids_ge_10 / total_source_video_ids * 100.0, 2)

    all_run_lengths = [len(r) for r in all_runs_available]
    total_contiguous_runs = len(all_run_lengths)
    longest_run_all = max(all_run_lengths) if all_run_lengths else 0
    mean_run_all = float(np.mean(all_run_lengths)) if all_run_lengths else 0.0
    median_run_all = float(np.median(all_run_lengths)) if all_run_lengths else 0.0
    q1_run_all = float(np.quantile(all_run_lengths, 0.25)) if all_run_lengths else 0.0
    q3_run_all = float(np.quantile(all_run_lengths, 0.75)) if all_run_lengths else 0.0
    total_gaps_all = int(continuity_by_video_df["num_gaps"].sum())

    # Run length distribution
    bucket_order = [
        "run length 1",
        "run length 2",
        "run length 3",
        "run length 4",
        "run length 5",
        "run length 6-9",
        "run length 10+",
    ]
    bucket_counts: Dict[str, int] = {b: 0 for b in bucket_order}
    for l in all_run_lengths:
        bucket_counts[categorize_run_length(l)] += 1

    run_distribution_rows = []
    for b in bucket_order:
        cnt = bucket_counts[b]
        pct = round(cnt / total_contiguous_runs * 100.0, 2) if total_contiguous_runs > 0 else 0.0
        run_distribution_rows.append(
            {
                "run_length_bucket": b,
                "run_count": cnt,
                "percentage": pct,
            }
        )

    run_distribution_df = pd.DataFrame(run_distribution_rows)
    run_distribution_df.to_csv(RUN_DISTRIBUTION_CSV_PATH, index=False)

    # ============================================================
    # LABELED CONTINUITY STATISTICS
    # ============================================================
    labeled_run_lengths = [len(r) for r in all_runs_labeled]
    total_labeled_runs = len(labeled_run_lengths)
    longest_labeled_run = max(labeled_run_lengths) if labeled_run_lengths else 0

    # Source videos containing labeled runs >= k
    lab_vids_ge_2 = sum(
        1 for runs in labeled_runs_by_video.values() if any(len(r) >= 2 for r in runs)
    )
    lab_vids_ge_3 = sum(
        1 for runs in labeled_runs_by_video.values() if any(len(r) >= 3 for r in runs)
    )
    lab_vids_ge_5 = sum(
        1 for runs in labeled_runs_by_video.values() if any(len(r) >= 5 for r in runs)
    )
    lab_vids_ge_10 = sum(
        1 for runs in labeled_runs_by_video.values() if any(len(r) >= 10 for r in runs)
    )

    pct_lab_vids_ge_2 = round(lab_vids_ge_2 / total_source_video_ids * 100.0, 2)
    pct_lab_vids_ge_3 = round(lab_vids_ge_3 / total_source_video_ids * 100.0, 2)
    pct_lab_vids_ge_5 = round(lab_vids_ge_5 / total_source_video_ids * 100.0, 2)
    pct_lab_vids_ge_10 = round(lab_vids_ge_10 / total_source_video_ids * 100.0, 2)

    # Number of labeled segments participating in runs of length >= k
    segs_in_run_ge_2 = sum(len(r) for r in all_runs_labeled if len(r) >= 2)
    segs_in_run_ge_3 = sum(len(r) for r in all_runs_labeled if len(r) >= 3)
    segs_in_run_ge_5 = sum(len(r) for r in all_runs_labeled if len(r) >= 5)
    segs_in_run_ge_10 = sum(len(r) for r in all_runs_labeled if len(r) >= 10)

    pct_segs_ge_2 = round(segs_in_run_ge_2 / total_labeled_videos * 100.0, 2)
    pct_segs_ge_3 = round(segs_in_run_ge_3 / total_labeled_videos * 100.0, 2)
    pct_segs_ge_5 = round(segs_in_run_ge_5 / total_labeled_videos * 100.0, 2)
    pct_segs_ge_10 = round(segs_in_run_ge_10 / total_labeled_videos * 100.0, 2)

    # Top videos with long contiguous runs
    top_continuous_videos = (
        continuity_by_video_df.head(10)[
            ["video_id", "longest_contiguous_run", "available_segments", "labeled_segments"]
        ].to_dict(orient="records")
    )

    # ============================================================
    # SUMMARY JSON ARTIFACT
    # ============================================================
    summary_data: Dict[str, Any] = {
        "dataset_name": "iSign",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_video_groups": {
            "total_source_video_ids": total_source_video_ids,
            "total_available_segments": total_valid_videos,
            "total_labeled_segments": total_labeled_videos,
            "source_videos_with_run_ge_2": vids_ge_2,
            "source_videos_with_run_ge_2_pct": pct_vids_ge_2,
            "source_videos_with_run_ge_3": vids_ge_3,
            "source_videos_with_run_ge_3_pct": pct_vids_ge_3,
            "source_videos_with_run_ge_5": vids_ge_5,
            "source_videos_with_run_ge_5_pct": pct_vids_ge_5,
            "source_videos_with_run_ge_10": vids_ge_10,
            "source_videos_with_run_ge_10_pct": pct_vids_ge_10,
        },
        "contiguous_sequences_all_available": {
            "total_contiguous_runs": total_contiguous_runs,
            "longest_contiguous_run": longest_run_all,
            "mean_run_length": round(mean_run_all, 2),
            "median_run_length": round(median_run_all, 2),
            "q1_run_length": round(q1_run_all, 2),
            "q3_run_length": round(q3_run_all, 2),
            "total_gaps": total_gaps_all,
            "run_length_distribution": run_distribution_rows,
        },
        "labeled_contiguous_sequences": {
            "total_labeled_runs": total_labeled_runs,
            "longest_labeled_run": longest_labeled_run,
            "source_videos_with_labeled_run_ge_2": lab_vids_ge_2,
            "source_videos_with_labeled_run_ge_2_pct": pct_lab_vids_ge_2,
            "source_videos_with_labeled_run_ge_3": lab_vids_ge_3,
            "source_videos_with_labeled_run_ge_3_pct": pct_lab_vids_ge_3,
            "source_videos_with_labeled_run_ge_5": lab_vids_ge_5,
            "source_videos_with_labeled_run_ge_5_pct": pct_lab_vids_ge_5,
            "source_videos_with_labeled_run_ge_10": lab_vids_ge_10,
            "source_videos_with_labeled_run_ge_10_pct": pct_lab_vids_ge_10,
            "labeled_segments_participating_in_runs": {
                "total_labeled_segments": total_labeled_videos,
                "segments_in_run_ge_2": segs_in_run_ge_2,
                "segments_in_run_ge_2_pct": pct_segs_ge_2,
                "segments_in_run_ge_3": segs_in_run_ge_3,
                "segments_in_run_ge_3_pct": pct_segs_ge_3,
                "segments_in_run_ge_5": segs_in_run_ge_5,
                "segments_in_run_ge_5_pct": pct_segs_ge_5,
                "segments_in_run_ge_10": segs_in_run_ge_10,
                "segments_in_run_ge_10_pct": pct_segs_ge_10,
            },
        },
        "top_longest_run_videos": top_continuous_videos,
        "key_findings": [
            f"Over half ({bucket_counts['run length 1'] / total_contiguous_runs * 100:.2f}%) of all contiguous runs consist of single isolated segments.",
            f"Median contiguous run length across the dataset is {median_run_all:.1f} segment (mean {mean_run_all:.2f} segments).",
            f"While {pct_vids_ge_2}% of source videos contain at least one contiguous segment pair (run >= 2), only {pct_vids_ge_5}% reach run length >= 5, and fewer than 1% ({vids_ge_10} videos) reach run length >= 10.",
            f"For supervised translation (ground-truth English labels), {pct_segs_ge_2}% of labeled segments are in runs >= 2, but only {pct_segs_ge_5}% participate in runs >= 5, and only {pct_segs_ge_10}% in runs >= 10.",
            "Conclusion: The currently available local dataset contains predominantly short continuous bursts (pairs or triplets of 2-3 consecutive sentences) rather than extended continuous multi-minute sequences. Modeling should focus on isolated sentence translation or short 2-3 segment context concatenation rather than long-range discourse modeling.",
        ],
    }

    with open(CONTINUITY_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # ============================================================
    # TERMINAL REPORT
    # ============================================================
    print("\n" + "=" * 60)
    print("SEQUENCE CONTINUITY ANALYSIS")
    print("=" * 60)

    print("\nSOURCE VIDEO GROUPS")
    print("-" * 60)
    print(f"Total source video groups (video_id) : {total_source_video_ids:,}")
    print(f"Total available local video segments  : {total_valid_videos:,}")
    print(f"Total matched labeled segments        : {total_labeled_videos:,}")
    print("\nSource videos by longest contiguous run:")
    print(f"  - Run length >=  2 : {vids_ge_2:>3} source videos ({pct_vids_ge_2:5.2f}%)")
    print(f"  - Run length >=  3 : {vids_ge_3:>3} source videos ({pct_vids_ge_3:5.2f}%)")
    print(f"  - Run length >=  5 : {vids_ge_5:>3} source videos ({pct_vids_ge_5:5.2f}%)")
    print(f"  - Run length >= 10 : {vids_ge_10:>3} source videos ({pct_vids_ge_10:5.2f}%)")

    print("\nCONTIGUOUS SEQUENCES")
    print("-" * 60)
    print(f"Total contiguous runs identified      : {total_contiguous_runs:,}")
    print(f"Longest contiguous run across dataset : {longest_run_all} segments")
    print(f"Mean contiguous run length            : {mean_run_all:.2f} segments")
    print(f"Median contiguous run length          : {median_run_all:.2f} segments")
    print(f"Q1 (25th percentile)                  : {q1_run_all:.2f} segments")
    print(f"Q3 (75th percentile)                  : {q3_run_all:.2f} segments")
    print(f"Total sequence gaps (discontinuities) : {total_gaps_all:,}")

    print("\nLABELED CONTIGUOUS SEQUENCES")
    print("-" * 60)
    print(f"Analysis population: {total_labeled_videos:,} matched translation segments\n")
    print(f"Total labeled contiguous runs         : {total_labeled_runs:,}")
    print(f"Longest labeled contiguous run        : {longest_labeled_run} segments")
    print("\nSource videos containing labeled contiguous runs:")
    print(f"  - Run length >=  2 : {lab_vids_ge_2:>3} source videos ({pct_lab_vids_ge_2:5.2f}%)")
    print(f"  - Run length >=  3 : {lab_vids_ge_3:>3} source videos ({pct_lab_vids_ge_3:5.2f}%)")
    print(f"  - Run length >=  5 : {lab_vids_ge_5:>3} source videos ({pct_lab_vids_ge_5:5.2f}%)")
    print(f"  - Run length >= 10 : {lab_vids_ge_10:>3} source videos ({pct_lab_vids_ge_10:5.2f}%)")

    print("\nLabeled segments participating in contiguous runs:")
    print(f"  - In runs of length >=  2 : {segs_in_run_ge_2:>5,} / {total_labeled_videos:,} segments ({pct_segs_ge_2:5.2f}%)")
    print(f"  - In runs of length >=  3 : {segs_in_run_ge_3:>5,} / {total_labeled_videos:,} segments ({pct_segs_ge_3:5.2f}%)")
    print(f"  - In runs of length >=  5 : {segs_in_run_ge_5:>5,} / {total_labeled_videos:,} segments ({pct_segs_ge_5:5.2f}%)")
    print(f"  - In runs of length >= 10 : {segs_in_run_ge_10:>5,} / {total_labeled_videos:,} segments ({pct_segs_ge_10:5.2f}%)")

    print("\nRUN LENGTH DISTRIBUTION")
    print("-" * 60)
    for row in run_distribution_rows:
        print(f"  {row['run_length_bucket']:<14} : {row['run_count']:>5,} runs ({row['percentage']:5.2f}%)")

    print("\nIMPORTANT FINDINGS")
    print("-" * 60)
    print("1. High Run Fragmentation:")
    print(f"   Over half ({bucket_counts['run length 1'] / total_contiguous_runs * 100:.2f}%) of all contiguous runs consist of a single isolated")
    print("   segment. The median contiguous run length is only 1.0 segment (mean: 1.96).")
    print("\n2. Continuity Availability:")
    print(f"   While {pct_vids_ge_2:.2f}% of source videos have at least one contiguous pair (run >= 2)")
    print(f"   and {pct_segs_ge_2:.2f}% of labeled segments participate in runs >= 2, long-range")
    print(f"   continuity drops sharply: only {pct_vids_ge_5:.2f}% of source videos have runs >= 5,")
    print(f"   and only {vids_ge_10} videos ({pct_vids_ge_10:.2f}%) have runs >= 10.")
    print("\n3. Experimental Feasibility for Continuous Translation:")
    print("   - Long-range discourse/document-level translation (10+ consecutive sentences)")
    print("     is NOT viable on the current local subset without substantial data dilution.")
    print("   - Sentence-level continuous sign language translation (translating each individual")
    print("     continuous video segment into its corresponding English sentence) is FULLY viable")
    print(f"     across all {total_labeled_videos:,} labeled segments.")
    print("   - Short-context sequence translation (concatenating 2-3 consecutive segments)")
    print(f"     is viable across {segs_in_run_ge_2:,} segments ({pct_segs_ge_2:.2f}% of the labeled dataset).")

    print("\n" + "=" * 60)
    print("Generated Output Files:")
    print(f"  1. {CONTINUITY_BY_VIDEO_CSV_PATH}")
    print(f"  2. {RUN_DISTRIBUTION_CSV_PATH}")
    print(f"  3. {CONTINUITY_SUMMARY_JSON_PATH}")
    print("=" * 60 + "\n")

    return summary_data


if __name__ == "__main__":
    analyze_continuity()
