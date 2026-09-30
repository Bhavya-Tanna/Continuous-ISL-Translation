import struct
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem


# ============================================================
# CONFIG
# ============================================================

REPO = "datasets/Exploration-Lab/iSign"

REMOTE_FILE = "iSign-videos_v1.1_part_aa"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "selected_video_index.csv"
)


# ============================================================
# LOAD CORRECTED INDEX
# ============================================================

df = pd.read_csv(INDEX_PATH)

target = df[
    df["filename"].str.contains(
        "W_LjNlwlqSI--10.mp4",
        regex=False
    )
]

if len(target) != 1:

    raise RuntimeError(
        f"Expected exactly one target entry, "
        f"found {len(target)}"
    )

row = target.iloc[0]

offset = int(
    row["local_header_offset"]
)

print("=" * 70)
print("ZIP64 OFFSET VERIFICATION")
print("=" * 70)

print(
    f"\nTarget file:"
    f"\n{row['filename']}"
)

print(
    f"\nCorrected local header offset:"
    f"\n{offset:,}"
)

print(
    f"\nCompressed size:"
    f"\n{int(row['compressed_size']):,}"
)


# ============================================================
# CONNECT
# ============================================================

print(
    "\nConnecting to Hugging Face..."
)

fs = HfFileSystem()

remote_path = (
    f"{REPO}/{REMOTE_FILE}"
)


# ============================================================
# READ ONLY 30 BYTES
# ============================================================

print(
    "\nRequesting 30 bytes from corrected offset..."
)

with fs.open(
    remote_path,
    "rb"
) as f:

    f.seek(offset)

    header = f.read(30)


# ============================================================
# SHOW RESULT
# ============================================================

print(
    "\nReceived bytes:"
)

print(
    header.hex(" ")
)

print(
    "\nFirst 4 bytes:"
)

print(
    header[:4]
)


# ============================================================
# VALIDATE ZIP SIGNATURE
# ============================================================

if header[:4] != b"PK\x03\x04":

    raise RuntimeError(
        "\nFAILED: corrected offset does NOT "
        "point to a ZIP local header."
    )


# ============================================================
# PARSE HEADER
# ============================================================

(
    signature,
    version,
    flags,
    compression,
    mod_time,
    mod_date,
    crc32,
    compressed_size,
    uncompressed_size,
    filename_length,
    extra_length,
) = struct.unpack(
    "<4s5H3I2H",
    header
)


print(
    "\n" + "=" * 70
)

print(
    "VERIFICATION SUCCESSFUL"
)

print(
    "=" * 70
)

print(
    "\n✓ ZIP signature: PK 03 04"
)

print(
    f"✓ Compression method: {compression}"
)

print(
    f"✓ Filename length: {filename_length}"
)

print(
    f"✓ Extra field length: {extra_length}"
)

print(
    "\nThe corrected ZIP64 offset points "
    "to a valid ZIP local header."
)

print(
    "\nSafe to resume selective video download."
)

print(
    "=" * 70
)