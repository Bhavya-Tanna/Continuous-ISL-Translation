import pandas as pd
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "iSign_v1.1.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

print("\n" + "=" * 60)
print("iSIGN METADATA PROFILING")
print("=" * 60)

print(f"\nTotal rows: {len(df):,}")


# ============================================================
# CLEAN TEXT
# ============================================================

df["text"] = df["text"].fillna("").astype(str).str.strip()

valid_df = df[df["text"] != ""].copy()

print(f"Valid text rows: {len(valid_df):,}")
print(f"Missing/empty text: {len(df) - len(valid_df):,}")


# ============================================================
# EXTRACT VIDEO ID
# ============================================================

valid_df["video_id"] = (
    valid_df["uid"]
    .str.rsplit("-", n=1)
    .str[0]
)

valid_df["sequence_number"] = (
    valid_df["uid"]
    .str.rsplit("-", n=1)
    .str[1]
)


# ============================================================
# UNIQUE VIDEOS
# ============================================================

unique_videos = valid_df["video_id"].nunique()

print("\n" + "-" * 60)
print("VIDEO STATISTICS")
print("-" * 60)

print(f"Unique video IDs: {unique_videos:,}")
print(
    f"Average segments per video: "
    f"{len(valid_df) / unique_videos:.2f}"
)


# ============================================================
# SEGMENTS PER VIDEO
# ============================================================

segments_per_video = (
    valid_df
    .groupby("video_id")
    .size()
)

print("\nSegments per video:")
print(segments_per_video.describe())


# ============================================================
# TEXT LENGTH
# ============================================================

valid_df["word_count"] = (
    valid_df["text"]
    .str.split()
    .str.len()
)

valid_df["character_count"] = (
    valid_df["text"]
    .str.len()
)

print("\n" + "-" * 60)
print("TEXT LENGTH")
print("-" * 60)

print("\nWord count:")
print(valid_df["word_count"].describe())

print("\nCharacter count:")
print(valid_df["character_count"].describe())


# ============================================================
# VOCABULARY
# ============================================================

def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())


all_words = []

for text in valid_df["text"]:
    all_words.extend(tokenize(text))

word_counts = Counter(all_words)

print("\n" + "-" * 60)
print("VOCABULARY")
print("-" * 60)

print(f"Total word occurrences: {len(all_words):,}")
print(f"Unique words: {len(word_counts):,}")

print("\nTop 30 words:")

for word, count in word_counts.most_common(30):
    print(f"{word:<20} {count:,}")


# ============================================================
# UNIQUE TRANSLATIONS
# ============================================================

unique_texts = valid_df["text"].nunique()

print("\n" + "-" * 60)
print("TRANSLATION DIVERSITY")
print("-" * 60)

print(f"Unique text translations: {unique_texts:,}")

duplicate_text_rows = len(valid_df) - unique_texts

print(
    f"Rows belonging to repeated translations: "
    f"{duplicate_text_rows:,}"
)


# ============================================================
# WORD-LENGTH DISTRIBUTION
# ============================================================

bins = [0, 5, 10, 15, 20, 30, 100]

labels = [
    "1-5 words",
    "6-10 words",
    "11-15 words",
    "16-20 words",
    "21-30 words",
    "31+ words"
]

valid_df["length_group"] = pd.cut(
    valid_df["word_count"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print("\n" + "-" * 60)
print("SENTENCE LENGTH DISTRIBUTION")
print("-" * 60)

print(
    valid_df["length_group"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 60)
print("PROFILING COMPLETE")
print("=" * 60)