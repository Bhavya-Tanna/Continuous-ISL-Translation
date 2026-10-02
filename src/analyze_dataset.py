"""
Continuous Indian Sign Language Translation Using Deep Learning
Dataset Characterization Analysis

Script: src/analyze_dataset.py

Objective:
    Characterize the actual usable dataset currently available locally.
    The primary analysis population is the 2,215 videos that passed video
    integrity validation. Metadata from selected_samples.csv is carefully joined
    using UID and video filename basenames to analyze text characteristics, video
    properties, source video distributions, and dataset coverage.

Outputs generated:
    1. data/metadata/dataset_summary.json
    2. data/metadata/dataset_characteristics.csv
    3. data/metadata/source_video_distribution.csv
    4. data/metadata/sentence_length_distribution.csv
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
SUMMARY_JSON_PATH = METADATA_DIR / "dataset_summary.json"
CHARACTERISTICS_CSV_PATH = METADATA_DIR / "dataset_characteristics.csv"
SOURCE_DIST_CSV_PATH = METADATA_DIR / "source_video_distribution.csv"
SENTENCE_LENGTH_CSV_PATH = METADATA_DIR / "sentence_length_distribution.csv"


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
        ("Archive Video Index", INDEX_PATH),
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
# DATA LOADING & PREPARATION
# ============================================================

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset metadata files, validate column structures, and derive
    standardized basename keys for robust joining across records.

    Relationship:
        selected_samples.uid -> uid + '.mp4' -> basename
        selected_video_index.filename -> basename
        video_integrity.csv.filename -> basename

    Returns:
        Tuple of (samples_df, index_df, integrity_df)
    """
    validate_input_files()

    # 1. Selected Samples
    samples_df = pd.read_csv(SAMPLES_PATH)
    validate_columns(
        samples_df,
        ["uid", "video_id", "sequence_number", "text", "word_count"],
        SAMPLES_PATH,
    )

    # 2. Selected Video Archive Index
    index_df = pd.read_csv(INDEX_PATH)
    validate_columns(
        index_df,
        ["filename", "video_id", "compressed_size", "uncompressed_size"],
        INDEX_PATH,
    )

    # 3. Video Integrity Report
    integrity_df = pd.read_csv(INTEGRITY_PATH)
    validate_columns(
        integrity_df,
        [
            "filename",
            "video_id",
            "status",
            "width",
            "height",
            "fps",
            "frame_count",
            "duration_seconds",
        ],
        INTEGRITY_PATH,
    )

    # Derive standardized basenames (e.g., "-CpDPjXeB2c--1.mp4")
    samples_df["basename"] = samples_df["uid"].astype(str).str.strip() + ".mp4"
    index_df["basename"] = index_df["filename"].apply(lambda p: Path(str(p)).name)
    integrity_df["basename"] = integrity_df["filename"].apply(
        lambda p: Path(str(p)).name
    )

    return samples_df, index_df, integrity_df


# ============================================================
# STATISTICAL HELPER FUNCTIONS
# ============================================================

def calc_quartiles(series: pd.Series) -> Dict[str, float]:
    """
    Calculate summary statistics including min, max, mean, median, Q1 (25%), Q3 (75%).
    """
    s = series.dropna().astype(float)
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }


# ============================================================
# ANALYSIS IMPLEMENTATION
# ============================================================

def analyze_dataset() -> Dict[str, Any]:
    """
    Perform full dataset characterization, compute statistical distributions,
    generate the 4 required output artifacts, and print the structured report.
    """
    samples_df, index_df, integrity_df = load_and_prepare_data()

    # ------------------------------------------------------------
    # 1. Primary Analysis Population Filter
    # ------------------------------------------------------------
    # Only videos that passed integrity validation form the usable dataset.
    valid_videos_df = integrity_df[integrity_df["status"] == "valid"].copy()

    total_selected_samples = len(samples_df)
    total_indexed_samples = len(index_df)
    total_valid_videos = len(valid_videos_df)

    # ------------------------------------------------------------
    # 2. Metadata Joins & Tracking
    # ------------------------------------------------------------
    # Samples in selected_samples that match the archive index
    samples_in_index = samples_df[samples_df["basename"].isin(index_df["basename"])]
    samples_represented_in_index_count = len(samples_in_index)

    # Samples that match valid videos in integrity report
    matched_samples_df = samples_df.merge(
        valid_videos_df,
        on="basename",
        how="inner",
        suffixes=("_sample", "_integrity"),
    )
    matched_samples_count = len(matched_samples_df)

    # Selected samples missing from downloadable archive portion
    missing_samples_df = samples_df[
        ~samples_df["basename"].isin(valid_videos_df["basename"])
    ]
    unavailable_selected_count = len(missing_samples_df)

    # Additional valid videos in archive not present in selected_samples
    additional_valid_videos_df = valid_videos_df[
        ~valid_videos_df["basename"].isin(samples_df["basename"])
    ]
    additional_valid_videos_count = len(additional_valid_videos_count_series := additional_valid_videos_df)

    # ------------------------------------------------------------
    # A. DATASET SIZE & SOURCE VIDEO GROUPS
    # ------------------------------------------------------------
    unique_source_video_ids = valid_videos_df["video_id"].nunique()
    segments_per_video_series = valid_videos_df.groupby("video_id").size()

    segments_stats = calc_quartiles(segments_per_video_series)

    # ------------------------------------------------------------
    # B. VIDEO PROPERTIES (From video_integrity.csv)
    # ------------------------------------------------------------
    valid_videos_df["resolution"] = (
        valid_videos_df["width"].astype(str)
        + "x"
        + valid_videos_df["height"].astype(str)
    )

    resolution_counts = valid_videos_df["resolution"].value_counts()
    fps_counts = valid_videos_df["fps"].value_counts().sort_index()

    fps_stats = calc_quartiles(valid_videos_df["fps"])
    frame_stats = calc_quartiles(valid_videos_df["frame_count"])
    duration_stats = calc_quartiles(valid_videos_df["duration_seconds"])

    total_duration_sec = float(valid_videos_df["duration_seconds"].sum())
    total_duration_min = total_duration_sec / 60.0
    total_duration_hr = total_duration_sec / 3600.0

    duration_stats["total_seconds"] = total_duration_sec
    duration_stats["total_minutes"] = total_duration_min
    duration_stats["total_hours"] = total_duration_hr

    # ------------------------------------------------------------
    # C. TEXT / TRANSLATION CHARACTERISTICS
    # ------------------------------------------------------------
    # Using the matched selected_samples rows
    total_translation_sentences = len(matched_samples_df)
    unique_translations = int(matched_samples_df["text"].nunique())
    duplicate_translations = total_translation_sentences - unique_translations

    word_count_stats = calc_quartiles(matched_samples_df["word_count"])

    # Sentence-length buckets:
    # 1–5 words, 6–10 words, 11–15 words, 16–20 words, 21–30 words, 31+ words
    bins = [0, 5, 10, 15, 20, 30, float("inf")]
    bucket_labels = [
        "1-5 words",
        "6-10 words",
        "11-15 words",
        "16-20 words",
        "21-30 words",
        "31+ words",
    ]
    matched_samples_df["length_bucket"] = pd.cut(
        matched_samples_df["word_count"],
        bins=bins,
        labels=bucket_labels,
        right=True,
    )
    bucket_counts_series = (
        matched_samples_df["length_bucket"]
        .value_counts()
        .reindex(bucket_labels, fill_value=0)
    )

    sentence_length_distribution_data = []
    bucket_ranges = [
        ("1-5 words", 1, 5),
        ("6-10 words", 6, 10),
        ("11-15 words", 11, 15),
        ("16-20 words", 16, 20),
        ("21-30 words", 21, 30),
        ("31+ words", 31, None),
    ]
    for label, min_w, max_w in bucket_ranges:
        cnt = int(bucket_counts_series[label])
        pct = (cnt / total_translation_sentences * 100.0) if total_translation_sentences > 0 else 0.0
        sentence_length_distribution_data.append(
            {
                "bucket": label,
                "min_words": min_w,
                "max_words": max_w if max_w is not None else "",
                "count": cnt,
                "percentage": round(pct, 2),
            }
        )

    # Vocabulary using normalized lowercase whitespace-tokenized English text
    norm_texts = (
        matched_samples_df["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    all_tokens = [word for sent in norm_texts for word in sent.split()]
    total_word_occurrences = len(all_tokens)
    vocabulary_size = len(set(all_tokens))

    char_counts_series = matched_samples_df["text"].fillna("").astype(str).str.len()
    char_count_stats = {
        "min": int(char_counts_series.min()),
        "max": int(char_counts_series.max()),
        "mean": float(char_counts_series.mean()),
        "median": float(char_counts_series.median()),
        "q1": float(char_counts_series.quantile(0.25)),
        "q3": float(char_counts_series.quantile(0.75)),
    }

    # ------------------------------------------------------------
    # D. SOURCE VIDEO DISTRIBUTION
    # ------------------------------------------------------------
    source_grouped = (
        valid_videos_df.groupby("video_id")
        .agg(
            segment_count=("filename", "count"),
            total_duration_seconds=("duration_seconds", "sum"),
            avg_duration_seconds=("duration_seconds", "mean"),
        )
        .reset_index()
    )

    source_grouped["total_duration_seconds"] = source_grouped[
        "total_duration_seconds"
    ].round(2)
    source_grouped["avg_duration_seconds"] = source_grouped[
        "avg_duration_seconds"
    ].round(2)

    source_grouped = source_grouped.sort_values(
        by=["segment_count", "video_id"], ascending=[False, True]
    ).reset_index(drop=True)

    segment_freq_series = source_grouped["segment_count"].value_counts().sort_index()

    top_10_source_videos = source_grouped.head(10).to_dict(orient="records")

    # ------------------------------------------------------------
    # E. DATASET COVERAGE / DROPPED SAMPLES
    # ------------------------------------------------------------
    pct_retained = (
        (matched_samples_count / total_selected_samples) * 100.0
        if total_selected_samples > 0
        else 0.0
    )
    pct_unavailable = (
        (unavailable_selected_count / total_selected_samples) * 100.0
        if total_selected_samples > 0
        else 0.0
    )
    pct_indexed_vs_selected = (
        (total_indexed_samples / total_selected_samples) * 100.0
        if total_selected_samples > 0
        else 0.0
    )

    coverage_explanation = (
        "The difference between the 3,998 originally selected metadata samples and the "
        "2,215 locally available valid videos is due to the safe archive subset strategy: "
        "only video entries fully contained within the downloaded initial archive portion "
        "(part_aa) were indexed and retrieved. The remaining 1,801 selected samples are "
        "located in unretrieved subsequent archive parts (part_ab onwards) and are not corrupt."
    )

    # ============================================================
    # BUILD DATA STRUCTURES FOR EXPORT
    # ============================================================

    # 1. Summary JSON dictionary
    summary_data = {
        "dataset_name": "iSign",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": {
            "total_selected_metadata_samples": total_selected_samples,
            "total_indexed_samples": total_indexed_samples,
            "total_valid_local_videos": total_valid_videos,
            "matched_samples_with_text": matched_samples_count,
            "unique_source_video_ids": unique_source_video_ids,
            "segments_per_video_id": {
                "min": int(segments_stats["min"]),
                "max": int(segments_stats["max"]),
                "mean": round(segments_stats["mean"], 4),
                "median": round(segments_stats["median"], 4),
                "q1": round(segments_stats["q1"], 4),
                "q3": round(segments_stats["q3"], 4),
            },
        },
        "dataset_coverage": {
            "selected_metadata_rows": total_selected_samples,
            "locally_indexed_rows": total_indexed_samples,
            "valid_local_videos": total_valid_videos,
            "matched_selected_rows": matched_samples_count,
            "unavailable_selected_rows": unavailable_selected_count,
            "additional_indexed_videos": additional_valid_videos_count,
            "percentage_retained": round(pct_retained, 2),
            "percentage_unavailable": round(pct_unavailable, 2),
            "percentage_indexed_vs_selected": round(pct_indexed_vs_selected, 2),
            "explanation": coverage_explanation,
        },
        "video_properties": {
            "total_valid_videos": total_valid_videos,
            "resolution": {
                "unique_resolutions_count": int(valid_videos_df["resolution"].nunique()),
                "most_common_resolutions": [
                    {
                        "resolution": str(res),
                        "count": int(cnt),
                        "percentage": round(cnt / total_valid_videos * 100.0, 2),
                    }
                    for res, cnt in resolution_counts.head(5).items()
                ],
            },
            "fps": {
                "min": round(fps_stats["min"], 2),
                "max": round(fps_stats["max"], 2),
                "mean": round(fps_stats["mean"], 2),
                "median": round(fps_stats["median"], 2),
                "q1": round(fps_stats["q1"], 2),
                "q3": round(fps_stats["q3"], 2),
                "distribution": {
                    f"{fps:.2f}": int(cnt) for fps, cnt in fps_counts.items()
                },
            },
            "frame_count": {
                "min": int(frame_stats["min"]),
                "max": int(frame_stats["max"]),
                "mean": round(frame_stats["mean"], 2),
                "median": round(frame_stats["median"], 2),
                "q1": round(frame_stats["q1"], 2),
                "q3": round(frame_stats["q3"], 2),
            },
            "duration_seconds": {
                "min": round(duration_stats["min"], 2),
                "max": round(duration_stats["max"], 2),
                "mean": round(duration_stats["mean"], 2),
                "median": round(duration_stats["median"], 2),
                "q1": round(duration_stats["q1"], 2),
                "q3": round(duration_stats["q3"], 2),
                "total_seconds": round(total_duration_sec, 2),
                "total_minutes": round(total_duration_min, 2),
                "total_hours": round(total_duration_hr, 2),
            },
        },
        "text_characteristics": {
            "analysis_population": matched_samples_count,
            "total_translation_sentences": total_translation_sentences,
            "unique_translations": unique_translations,
            "duplicate_translation_count": duplicate_translations,
            "word_count": {
                "min": int(word_count_stats["min"]),
                "max": int(word_count_stats["max"]),
                "mean": round(word_count_stats["mean"], 2),
                "median": round(word_count_stats["median"], 2),
                "q1": round(word_count_stats["q1"], 2),
                "q3": round(word_count_stats["q3"], 2),
            },
            "character_count": {
                "min": int(char_count_stats["min"]),
                "max": int(char_count_stats["max"]),
                "mean": round(char_count_stats["mean"], 2),
                "median": round(char_count_stats["median"], 2),
                "q1": round(char_count_stats["q1"], 2),
                "q3": round(char_count_stats["q3"], 2),
            },
            "vocabulary": {
                "total_word_occurrences": total_word_occurrences,
                "vocabulary_size": vocabulary_size,
            },
            "sentence_length_distribution": sentence_length_distribution_data,
        },
        "source_video_distribution": {
            "unique_source_video_ids": unique_source_video_ids,
            "signer_identity_note": (
                "Source video IDs represent video recording groups and do NOT "
                "indicate signer identity (signer ID is not provided in metadata)."
            ),
            "segment_frequency_distribution": {
                f"{int(k)}": int(v) for k, v in segment_freq_series.items()
            },
            "top_10_source_videos": top_10_source_videos,
        },
    }

    # 2. Dataset Characteristics CSV table
    characteristics_rows = [
        # Dataset Size
        ("Dataset Size", "Total Selected Metadata Samples", total_selected_samples),
        ("Dataset Size", "Total Indexed Archive Samples", total_indexed_samples),
        ("Dataset Size", "Total Valid Local Videos", total_valid_videos),
        ("Dataset Size", "Matched Text-Video Pairs", matched_samples_count),
        ("Dataset Size", "Unique Source Video IDs", unique_source_video_ids),
        ("Dataset Size", "Min Segments per Video ID", int(segments_stats["min"])),
        ("Dataset Size", "Max Segments per Video ID", int(segments_stats["max"])),
        ("Dataset Size", "Mean Segments per Video ID", round(segments_stats["mean"], 4)),
        ("Dataset Size", "Median Segments per Video ID", round(segments_stats["median"], 2)),
        ("Dataset Size", "Q1 Segments per Video ID", round(segments_stats["q1"], 2)),
        ("Dataset Size", "Q3 Segments per Video ID", round(segments_stats["q3"], 2)),
        # Coverage
        ("Coverage", "Selected Metadata Rows", total_selected_samples),
        ("Coverage", "Locally Indexed Rows", total_indexed_samples),
        ("Coverage", "Valid Local Videos", total_valid_videos),
        ("Coverage", "Unavailable Selected Rows", unavailable_selected_count),
        ("Coverage", "Additional Indexed Videos", additional_valid_videos_count),
        ("Coverage", "Percentage Retained (%)", round(pct_retained, 2)),
        ("Coverage", "Percentage Unavailable (%)", round(pct_unavailable, 2)),
        ("Coverage", "Percentage Indexed vs Selected (%)", round(pct_indexed_vs_selected, 2)),
        # Video Properties
        ("Video Properties", "Total Valid Videos", total_valid_videos),
        ("Video Properties", "Unique Resolutions Count", int(valid_videos_df["resolution"].nunique())),
        ("Video Properties", "Dominant Resolution", f"300x300 ({resolution_counts.get('300x300', 0)} videos, {round(resolution_counts.get('300x300', 0)/total_valid_videos*100, 2)}%)"),
        ("Video Properties", "Min FPS", round(fps_stats["min"], 2)),
        ("Video Properties", "Max FPS", round(fps_stats["max"], 2)),
        ("Video Properties", "Mean FPS", round(fps_stats["mean"], 2)),
        ("Video Properties", "Median FPS", round(fps_stats["median"], 2)),
        ("Video Properties", "Q1 FPS", round(fps_stats["q1"], 2)),
        ("Video Properties", "Q3 FPS", round(fps_stats["q3"], 2)),
        ("Video Properties", "Min Frame Count", int(frame_stats["min"])),
        ("Video Properties", "Max Frame Count", int(frame_stats["max"])),
        ("Video Properties", "Mean Frame Count", round(frame_stats["mean"], 2)),
        ("Video Properties", "Median Frame Count", round(frame_stats["median"], 2)),
        ("Video Properties", "Q1 Frame Count", round(frame_stats["q1"], 2)),
        ("Video Properties", "Q3 Frame Count", round(frame_stats["q3"], 2)),
        ("Video Properties", "Min Duration (seconds)", round(duration_stats["min"], 2)),
        ("Video Properties", "Max Duration (seconds)", round(duration_stats["max"], 2)),
        ("Video Properties", "Mean Duration (seconds)", round(duration_stats["mean"], 2)),
        ("Video Properties", "Median Duration (seconds)", round(duration_stats["median"], 2)),
        ("Video Properties", "Q1 Duration (seconds)", round(duration_stats["q1"], 2)),
        ("Video Properties", "Q3 Duration (seconds)", round(duration_stats["q3"], 2)),
        ("Video Properties", "Total Duration (seconds)", round(total_duration_sec, 2)),
        ("Video Properties", "Total Duration (minutes)", round(total_duration_min, 2)),
        ("Video Properties", "Total Duration (hours)", round(total_duration_hr, 2)),
        # Text Characteristics
        ("Text Characteristics", "Total Translation Sentences", total_translation_sentences),
        ("Text Characteristics", "Unique Translations", unique_translations),
        ("Text Characteristics", "Duplicate Translations", duplicate_translations),
        ("Text Characteristics", "Total Word Occurrences", total_word_occurrences),
        ("Text Characteristics", "Vocabulary Size", vocabulary_size),
        ("Text Characteristics", "Min Word Count", int(word_count_stats["min"])),
        ("Text Characteristics", "Max Word Count", int(word_count_stats["max"])),
        ("Text Characteristics", "Mean Word Count", round(word_count_stats["mean"], 2)),
        ("Text Characteristics", "Median Word Count", round(word_count_stats["median"], 2)),
        ("Text Characteristics", "Q1 Word Count", round(word_count_stats["q1"], 2)),
        ("Text Characteristics", "Q3 Word Count", round(word_count_stats["q3"], 2)),
        ("Text Characteristics", "Min Character Count", int(char_count_stats["min"])),
        ("Text Characteristics", "Max Character Count", int(char_count_stats["max"])),
        ("Text Characteristics", "Mean Character Count", round(char_count_stats["mean"], 2)),
        ("Text Characteristics", "Median Character Count", round(char_count_stats["median"], 2)),
    ]
    characteristics_df = pd.DataFrame(
        characteristics_rows, columns=["category", "metric", "value"]
    )

    # 3. Source Video Distribution CSV
    # source_grouped has: video_id, segment_count, total_duration_seconds, avg_duration_seconds

    # 4. Sentence Length Distribution CSV
    sentence_length_df = pd.DataFrame(sentence_length_distribution_data)

    # ============================================================
    # SAVE OUTPUT ARTIFACTS
    # ============================================================
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    characteristics_df.to_csv(CHARACTERISTICS_CSV_PATH, index=False)
    source_grouped.to_csv(SOURCE_DIST_CSV_PATH, index=False)
    sentence_length_df.to_csv(SENTENCE_LENGTH_CSV_PATH, index=False)

    # ============================================================
    # CONSTRUCT TERMINAL REPORT
    # ============================================================
    print("\n" + "=" * 60)
    print("DATASET CHARACTERIZATION")
    print("=" * 60)

    print("\nDATASET SIZE")
    print("-" * 60)
    print(f"Total selected metadata samples : {total_selected_samples:,}")
    print(f"Total indexed archive samples   : {total_indexed_samples:,}")
    print(f"Total valid local videos        : {total_valid_videos:,}")
    print(f"Matched text-video pairs        : {matched_samples_count:,}")
    print(f"Total source video groups       : {unique_source_video_ids:,}")
    print("\nSegments per source video (Valid Videos):")
    print(f"  Minimum : {int(segments_stats['min'])}")
    print(f"  Maximum : {int(segments_stats['max'])}")
    print(f"  Mean    : {segments_stats['mean']:.2f}")
    print(f"  Median  : {segments_stats['median']:.2f}")
    print(f"  Q1 (25%): {segments_stats['q1']:.2f}")
    print(f"  Q3 (75%): {segments_stats['q3']:.2f}")

    print("\nVIDEO CHARACTERISTICS")
    print("-" * 60)
    print(f"Analysis population: {total_valid_videos:,} integrity-validated videos\n")
    print("Resolution Distribution:")
    print(f"  Unique resolutions count : {valid_videos_df['resolution'].nunique()}")
    print("  Most common resolutions :")
    for res, cnt in resolution_counts.head(5).items():
        pct = cnt / total_valid_videos * 100.0
        print(f"    - {res:<10} : {cnt:>5,} videos ({pct:6.2f}%)")

    print("\nFrame Rate (FPS) Distribution:")
    for fps, cnt in fps_counts.items():
        pct = cnt / total_valid_videos * 100.0
        print(f"    - {fps:5.2f} FPS : {cnt:>5,} videos ({pct:6.2f}%)")

    print("\nFPS Statistics:")
    print(f"  Minimum : {fps_stats['min']:.2f}")
    print(f"  Maximum : {fps_stats['max']:.2f}")
    print(f"  Mean    : {fps_stats['mean']:.2f}")
    print(f"  Median  : {fps_stats['median']:.2f}")
    print(f"  Q1 (25%): {fps_stats['q1']:.2f}")
    print(f"  Q3 (75%): {fps_stats['q3']:.2f}")

    print("\nFrame Count Statistics:")
    print(f"  Minimum : {int(frame_stats['min']):,} frames")
    print(f"  Maximum : {int(frame_stats['max']):,} frames")
    print(f"  Mean    : {frame_stats['mean']:.2f} frames")
    print(f"  Median  : {frame_stats['median']:.2f} frames")
    print(f"  Q1 (25%): {frame_stats['q1']:.2f} frames")
    print(f"  Q3 (75%): {frame_stats['q3']:.2f} frames")

    print("\nVideo Duration Statistics:")
    print(f"  Minimum : {duration_stats['min']:.2f} seconds")
    print(f"  Maximum : {duration_stats['max']:.2f} seconds")
    print(f"  Mean    : {duration_stats['mean']:.2f} seconds")
    print(f"  Median  : {duration_stats['median']:.2f} seconds")
    print(f"  Q1 (25%): {duration_stats['q1']:.2f} seconds")
    print(f"  Q3 (75%): {duration_stats['q3']:.2f} seconds")
    print(f"  Total   : {total_duration_sec:,.2f} seconds "
          f"({total_duration_min:,.2f} minutes / {total_duration_hr:.2f} hours)")

    print("\nTEXT CHARACTERISTICS")
    print("-" * 60)
    print(f"Analysis population: {matched_samples_count:,} matched translation sentences\n")
    print("Sentence & Translation Statistics:")
    print(f"  Total translation sentences : {total_translation_sentences:,}")
    print(f"  Unique translations         : {unique_translations:,}")
    print(f"  Duplicate translations      : {duplicate_translations:,}")

    print("\nWord Count Statistics:")
    print(f"  Minimum : {int(word_count_stats['min'])} words")
    print(f"  Maximum : {int(word_count_stats['max'])} words")
    print(f"  Mean    : {word_count_stats['mean']:.2f} words")
    print(f"  Median  : {word_count_stats['median']:.2f} words")
    print(f"  Q1 (25%): {word_count_stats['q1']:.2f} words")
    print(f"  Q3 (75%): {word_count_stats['q3']:.2f} words")

    print("\nSentence-Length Distribution:")
    for row in sentence_length_distribution_data:
        print(f"  {row['bucket']:<11} : {row['count']:>5,} ({row['percentage']:5.2f}%)")

    print("\nVocabulary & Token Statistics:")
    print(f"  Total word occurrences (tokens) : {total_word_occurrences:,}")
    print(f"  Vocabulary size (unique tokens) : {vocabulary_size:,}")

    print("\nCharacter Count Statistics:")
    print(f"  Minimum : {char_count_stats['min']} characters")
    print(f"  Maximum : {char_count_stats['max']} characters")
    print(f"  Mean    : {char_count_stats['mean']:.2f} characters")
    print(f"  Median  : {char_count_stats['median']:.2f} characters")

    print("\nSOURCE VIDEO DISTRIBUTION")
    print("-" * 60)
    print(f"Unique source video IDs : {unique_source_video_ids:,}")
    print("Note: Source video IDs represent source video groups and do NOT indicate signer identity.\n")

    print("Distribution of segments per source video:")
    for seg_count, num_vids in segment_freq_series.items():
        label = "segment " if seg_count == 1 else "segments"
        print(f"  {seg_count:>2} {label} : {num_vids:>3} source videos")

    print("\nTop 10 Source Videos by Available Segments:")
    for idx, item in enumerate(top_10_source_videos, start=1):
        print(f"  {idx:>2}. {item['video_id']:<15} : {item['segment_count']:>2} segments "
              f"({item['total_duration_seconds']:>6.2f}s total)")

    print("\nDATASET COVERAGE")
    print("-" * 60)
    print(f"Original selected metadata rows : {total_selected_samples:>5,} (100.00%)")
    print(f"Locally indexed archive entries : {total_indexed_samples:>5,} ({pct_indexed_vs_selected:6.2f}%)")
    print(f"Valid local videos              : {total_valid_videos:>5,} ({pct_indexed_vs_selected:6.2f}%)")
    print(f"Matched usable selected rows    : {matched_samples_count:>5,} ({pct_retained:6.2f}%)")
    print(f"Unavailable selected rows       : {unavailable_selected_count:>5,} ({pct_unavailable:6.2f}%)")
    print(f"Additional local valid videos   : {additional_valid_videos_count:>5,} "
          f"({additional_valid_videos_count/total_valid_videos*100:6.2f}% of local videos)")

    print("\nCoverage Rationale:")
    print(f"  {coverage_explanation}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Valid videos                    : {total_valid_videos:,}")
    print(f"Unique source video_ids         : {unique_source_video_ids:,}")
    print(f"Total duration                  : {total_duration_sec:,.2f} s ({total_duration_hr:.2f} hrs)")
    print(f"Average duration                : {duration_stats['mean']:.2f} s")
    print(f"Median duration                 : {duration_stats['median']:.2f} s")
    dominant_res = resolution_counts.index[0]
    dominant_pct = resolution_counts.iloc[0] / total_valid_videos * 100.0
    print(f"Resolution(s)                   : {dominant_res} ({dominant_pct:.2f}%), {valid_videos_df['resolution'].nunique()} unique resolutions")
    print(f"FPS statistics                  : Mean: {fps_stats['mean']:.2f}, Median: {fps_stats['median']:.2f}, Range: [{fps_stats['min']:.2f}, {fps_stats['max']:.2f}]")
    print(f"Average sentence length         : {word_count_stats['mean']:.2f} words")
    print(f"Vocabulary size                 : {vocabulary_size:,} words")
    print(f"Percentage of selected retained : {pct_retained:.2f}% ({matched_samples_count:,} / {total_selected_samples:,} matched)")
    print("=" * 60)

    print("\nGenerated Output Files:")
    print(f"  1. {SUMMARY_JSON_PATH}")
    print(f"  2. {CHARACTERISTICS_CSV_PATH}")
    print(f"  3. {SOURCE_DIST_CSV_PATH}")
    print(f"  4. {SENTENCE_LENGTH_CSV_PATH}")
    print()

    return summary_data


if __name__ == "__main__":
    analyze_dataset()
