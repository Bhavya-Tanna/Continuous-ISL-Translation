import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "selected_video_index.csv"
)

PART_A_SIZE = 32_212_254_720


print("=" * 60)
print("SELECTED iSIGN VIDEO SIZE ANALYSIS")
print("=" * 60)


df = pd.read_csv(INDEX_PATH)


print(f"\nArchive entries: {len(df):,}")
print(
    f"Unique video IDs: "
    f"{df['video_id'].nunique():,}"
)


# ------------------------------------------------------------
# Total sizes
# ------------------------------------------------------------

compressed_total = df["compressed_size"].sum()
uncompressed_total = df["uncompressed_size"].sum()


print("\n" + "-" * 60)
print("TOTAL DATA")
print("-" * 60)

print(
    f"Compressed size   : "
    f"{compressed_total:,} bytes"
)

print(
    f"Compressed size   : "
    f"{compressed_total / (1024 ** 3):.2f} GB"
)

print(
    f"Uncompressed size : "
    f"{uncompressed_total:,} bytes"
)

print(
    f"Uncompressed size : "
    f"{uncompressed_total / (1024 ** 3):.2f} GB"
)


# ------------------------------------------------------------
# Determine which archive part contains each entry
# ------------------------------------------------------------

df["archive_part"] = df[
    "local_header_offset"
].apply(
    lambda x: (
        "part_aa"
        if x < PART_A_SIZE
        else "part_ab"
    )
)


print("\n" + "-" * 60)
print("ARCHIVE DISTRIBUTION")
print("-" * 60)

for part, group in df.groupby("archive_part"):

    compressed = group[
        "compressed_size"
    ].sum()

    uncompressed = group[
        "uncompressed_size"
    ].sum()

    print(f"\n{part}")

    print(
        f"  Entries       : {len(group):,}"
    )

    print(
        f"  Compressed    : "
        f"{compressed / (1024 ** 3):.2f} GB"
    )

    print(
        f"  Uncompressed  : "
        f"{uncompressed / (1024 ** 3):.2f} GB"
    )


# ------------------------------------------------------------
# Per-video totals
# ------------------------------------------------------------

video_sizes = (
    df.groupby("video_id")
    .agg(
        segments=("filename", "count"),
        compressed_bytes=("compressed_size", "sum"),
        uncompressed_bytes=("uncompressed_size", "sum")
    )
    .reset_index()
)


print("\n" + "-" * 60)
print("PER-SOURCE-VIDEO SIZE")
print("-" * 60)

print(
    f"Average compressed/video: "
    f"{video_sizes['compressed_bytes'].mean() / (1024 ** 2):.2f} MB"
)

print(
    f"Median compressed/video: "
    f"{video_sizes['compressed_bytes'].median() / (1024 ** 2):.2f} MB"
)

print(
    f"Largest compressed/video: "
    f"{video_sizes['compressed_bytes'].max() / (1024 ** 2):.2f} MB"
)

print(
    f"Smallest compressed/video: "
    f"{video_sizes['compressed_bytes'].min() / (1024 ** 2):.2f} MB"
)


# ------------------------------------------------------------
# Size percentiles
# ------------------------------------------------------------

print("\nCompressed size per source video:")

print(
    (
        video_sizes["compressed_bytes"]
        / (1024 ** 2)
    ).describe()
)


print("\n" + "=" * 60)
print("SIZE ANALYSIS COMPLETE")
print("=" * 60)