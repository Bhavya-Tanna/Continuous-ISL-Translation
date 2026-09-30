import io
import struct
import zlib
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem


# ============================================================
# CONFIG
# ============================================================

REPO = "datasets/Exploration-Lab/iSign"
REMOTE_FILE = "iSign-videos_v1.1_part_aa"

PART_A_SIZE = 32_212_254_720

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "selected_video_index.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "test_download"
)


# ============================================================
# LOAD INDEX
# ============================================================

df = pd.read_csv(INDEX_PATH)

# Use the first selected archive entry.
entry = df.iloc[0]

filename = entry["filename"]
video_id = entry["video_id"]

local_header_offset = int(
    entry["local_header_offset"]
)

compressed_size = int(
    entry["compressed_size"]
)

compression = int(
    entry["compression"]
)


print("=" * 60)
print("iSIGN SINGLE VIDEO RETRIEVAL TEST")
print("=" * 60)

print(f"\nVideo ID       : {video_id}")
print(f"Archive file   : {filename}")
print(f"Compression    : {compression}")
print(
    f"Compressed size: "
    f"{compressed_size:,} bytes"
)
print(
    f"Archive offset : "
    f"{local_header_offset:,}"
)


# ============================================================
# CONNECT
# ============================================================

fs = HfFileSystem()

remote_path = (
    f"{REPO}/{REMOTE_FILE}"
)


# ============================================================
# READ ZIP LOCAL HEADER
# ============================================================

LOCAL_HEADER_SIZE = 30

print("\nReading local ZIP header...")

with fs.open(remote_path, "rb") as f:

    f.seek(local_header_offset)

    local_header = f.read(
        LOCAL_HEADER_SIZE
    )


if len(local_header) != LOCAL_HEADER_SIZE:

    raise RuntimeError(
        "Could not read complete local header."
    )


# ZIP local file header signature
if local_header[:4] != b"PK\x03\x04":

    raise RuntimeError(
        "Invalid ZIP local file header."
    )


(
    signature,
    version,
    flags,
    compression_method,
    mod_time,
    mod_date,
    crc32,
    local_compressed_size,
    local_uncompressed_size,
    filename_length,
    extra_length,
) = struct.unpack(
    "<4s5H3I2H",
    local_header
)


print(
    f"Filename length: {filename_length}"
)

print(
    f"Extra field length: {extra_length}"
)

print(
    f"Compression method: "
    f"{compression_method}"
)


# ============================================================
# CALCULATE ACTUAL DATA OFFSET
# ============================================================

data_offset = (
    local_header_offset
    + LOCAL_HEADER_SIZE
    + filename_length
    + extra_length
)


print(
    f"\nActual compressed-data offset:"
    f"\n{data_offset:,}"
)


# ============================================================
# DOWNLOAD ONLY COMPRESSED VIDEO DATA
# ============================================================

print(
    f"\nRequesting only "
    f"{compressed_size:,} bytes..."
)

with fs.open(remote_path, "rb") as f:

    f.seek(data_offset)

    compressed_data = f.read(
        compressed_size
    )


print(
    f"Received: "
    f"{len(compressed_data):,} bytes"
)


if len(compressed_data) != compressed_size:

    raise RuntimeError(
        "Compressed data size mismatch."
    )


# ============================================================
# DECOMPRESS ZIP DEFLATE
# ============================================================

print("\nDecompressing video...")

if compression_method == 0:

    video_data = compressed_data

elif compression_method == 8:

    video_data = zlib.decompress(
        compressed_data,
        -15
    )

else:

    raise RuntimeError(
        f"Unsupported ZIP compression method: "
        f"{compression_method}"
    )


print(
    f"Decompressed size: "
    f"{len(video_data):,} bytes"
)


# ============================================================
# SAVE MP4
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    OUTPUT_DIR
    / Path(filename).name
)

with open(output_path, "wb") as f:

    f.write(video_data)


print(
    f"\nSaved video to:\n"
    f"{output_path}"
)


# ============================================================
# BASIC MP4 CHECK
# ============================================================

print("\n" + "-" * 60)
print("BASIC VIDEO CHECK")
print("-" * 60)

with open(output_path, "rb") as f:

    first_bytes = f.read(32)

print(
    "First 32 bytes:",
    first_bytes.hex(" ")
)

if b"ftyp" in first_bytes:

    print("MP4 container signature: FOUND")

else:

    print(
        "WARNING: MP4 signature not found."
    )


print("\n" + "=" * 60)
print("SINGLE VIDEO TEST COMPLETE")
print("=" * 60)
