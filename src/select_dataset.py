import pandas as pd
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "iSign_v1.1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)

SELECTED_SAMPLES_PATH = (
    OUTPUT_DIR
    / "selected_samples.csv"
)

SELECTED_VIDEOS_PATH = (
    OUTPUT_DIR
    / "selected_video_ids.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_SAMPLES = 4000

# We want reasonably sized source videos for our first dataset.
MIN_SEGMENTS_PER_VIDEO = 5
MAX_SEGMENTS_PER_VIDEO = 50

# Avoid extremely short/noisy translations.
MIN_WORDS = 3
MAX_WORDS = 25


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("iSIGN DATASET SELECTION")
print("=" * 60)

df = pd.read_csv(METADATA_PATH)

print(f"\nOriginal rows: {len(df):,}")


# ============================================================
# BASIC CLEANING
# ============================================================

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["text"] != ""].copy()

print(f"After removing missing text: {len(df):,}")


# ============================================================
# EXTRACT VIDEO ID + SEQUENCE NUMBER
# ============================================================

uid_parts = df["uid"].str.rsplit("-", n=1)

df["video_id"] = uid_parts.str[0]
df["sequence_number"] = pd.to_numeric(
    uid_parts.str[1],
    errors="coerce"
)

df = df.dropna(subset=["sequence_number"]).copy()

df["sequence_number"] = (
    df["sequence_number"]
    .astype(int)
)


# ============================================================
# TEXT NORMALIZATION FOR QUALITY CHECKING
# ============================================================

df["normalized_text"] = (
    df["text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["word_count"] = (
    df["normalized_text"]
    .str.split()
    .str.len()
)

df["character_count"] = (
    df["normalized_text"]
    .str.len()
)


# ============================================================
# REMOVE OBVIOUSLY NOISY TEXT
# ============================================================

def is_reasonable_text(text):
    """
    Keep normal English sentence/phrase text.
    Reject obvious metadata/page-number style entries.
    """

    # Must contain at least one alphabetic character
    if not re.search(r"[a-zA-Z]", text):
        return False

    # Reject entries that are almost entirely numbers
    if re.fullmatch(r"[\d\s.,:;()\-]+", text):
        return False

    # Reject obvious page-number style text
    if re.fullmatch(
        r"(page|pg\.?)\s*\d+",
        text,
        flags=re.IGNORECASE
    ):
        return False

    return True


before_quality = len(df)

df = df[
    df["text"].apply(is_reasonable_text)
].copy()

print(
    f"After basic text-quality filtering: "
    f"{len(df):,}"
)

print(
    f"Removed: {before_quality - len(df):,}"
)


# ============================================================
# SENTENCE LENGTH FILTER
# ============================================================

before_length = len(df)

df = df[
    (df["word_count"] >= MIN_WORDS) &
    (df["word_count"] <= MAX_WORDS)
].copy()

print(
    f"After sentence-length filtering: "
    f"{len(df):,}"
)

print(
    f"Removed: {before_length - len(df):,}"
)


# ============================================================
# VIDEO-LEVEL STATISTICS
# ============================================================

video_stats = (
    df
    .groupby("video_id")
    .agg(
        segment_count=("uid", "count"),
        avg_words=("word_count", "mean"),
        unique_texts=("normalized_text", "nunique")
    )
    .reset_index()
)

# Only consider source videos with a reasonable number of segments.
video_stats = video_stats[
    (video_stats["segment_count"] >= MIN_SEGMENTS_PER_VIDEO) &
    (video_stats["segment_count"] <= MAX_SEGMENTS_PER_VIDEO)
].copy()

print(
    f"\nCandidate source videos: "
    f"{len(video_stats):,}"
)


# ============================================================
# JOIN VIDEO STATS BACK TO SEGMENTS
# ============================================================

df = df.merge(
    video_stats[
        [
            "video_id",
            "segment_count",
            "avg_words",
            "unique_texts"
        ]
    ],
    on="video_id",
    how="inner"
)


# ============================================================
# VIDEO QUALITY SCORE
# ============================================================

# We don't want extremely short or extremely long videos
# to dominate the first dataset.

def length_balance_score(avg_words):

    if 7 <= avg_words <= 15:
        return 1.0

    if 5 <= avg_words < 7:
        return 0.8

    if 15 < avg_words <= 18:
        return 0.8

    if 3 <= avg_words < 5:
        return 0.6

    return 0.5


df["length_score"] = (
    df["avg_words"]
    .apply(length_balance_score)
)

df["diversity_score"] = (
    df["unique_texts"] /
    df["segment_count"]
)

df["video_score"] = (
    0.6 * df["length_score"] +
    0.4 * df["diversity_score"]
)


# ============================================================
# SORT SEGMENTS WITHIN EACH VIDEO
# ============================================================

df = df.sort_values(
    [
        "video_score",
        "video_id",
        "sequence_number"
    ],
    ascending=[False, True, True]
)


# ============================================================
# SELECT VIDEOS
# ============================================================

selected_video_ids = []
selected_count = 0

# Work with one row per video.
video_order = (
    df[
        [
            "video_id",
            "segment_count",
            "video_score"
        ]
    ]
    .drop_duplicates("video_id")
    .sort_values(
        [
            "video_score",
            "segment_count"
        ],
        ascending=[False, True]
    )
)

for _, row in video_order.iterrows():

    video_id = row["video_id"]
    segment_count = int(row["segment_count"])

    if selected_count + segment_count > TARGET_SAMPLES:
        continue

    selected_video_ids.append(video_id)
    selected_count += segment_count

    if selected_count >= TARGET_SAMPLES:
        break


# ============================================================
# EXTRACT ALL SELECTED SEGMENTS
# ============================================================

selected_df = df[
    df["video_id"].isin(selected_video_ids)
].copy()

selected_df = selected_df.sort_values(
    [
        "video_id",
        "sequence_number"
    ]
)

# Keep only useful columns.
selected_df = selected_df[
    [
        "uid",
        "video_id",
        "sequence_number",
        "text",
        "word_count"
    ]
]


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

selected_df.to_csv(
    SELECTED_SAMPLES_PATH,
    index=False
)

selected_videos_df = (
    selected_df[
        ["video_id"]
    ]
    .drop_duplicates()
    .sort_values("video_id")
)

selected_videos_df.to_csv(
    SELECTED_VIDEOS_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "-" * 60)
print("SELECTION RESULT")
print("-" * 60)

print(
    f"Selected samples : {len(selected_df):,}"
)

print(
    f"Selected videos  : "
    f"{selected_df['video_id'].nunique():,}"
)

print(
    f"Average segments/video : "
    f"{selected_df.groupby('video_id').size().mean():.2f}"
)

print(
    f"Unique translations : "
    f"{selected_df['text'].nunique():,}"
)

print("\nSelected sentence lengths:")

print(
    selected_df["word_count"]
    .describe()
)

print("\nFiles created:")

print(
    f"1. {SELECTED_SAMPLES_PATH}"
)

print(
    f"2. {SELECTED_VIDEOS_PATH}"
)

print("\n" + "=" * 60)
print("DATASET SELECTION COMPLETE")
print("=" * 60)