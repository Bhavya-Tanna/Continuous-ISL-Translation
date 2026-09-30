import struct
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "selected_video_ids.csv"
)

CACHE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "zip_central_directory.bin"
)

selected = pd.read_csv(SELECTED)

selected_ids = (
    selected["video_id"]
    .astype(str)
    .str.strip()
    .tolist()
)

print("\nSELECTED VIDEO IDs:")
for x in selected_ids[:10]:
    print(repr(x))

data = CACHE.read_bytes()

p = 0
names = []

while p + 46 <= len(data):

    if data[p:p+4] != b"PK\x01\x02":
        break

    header = data[p:p+46]

    (
        signature,
        version_made,
        version_needed,
        flags,
        compression,
        mod_time,
        mod_date,
        crc32,
        compressed_size,
        uncompressed_size,
        filename_length,
        extra_length,
        comment_length,
        disk_number,
        internal_attributes,
        external_attributes,
        local_header_offset,
    ) = struct.unpack(
        "<4s6H3I5H2I",
        header
    )

    filename_start = p + 46
    filename_end = filename_start + filename_length

    filename = data[
        filename_start:filename_end
    ].decode(
        "utf-8",
        errors="replace"
    )

    names.append(
        Path(filename).name
    )

    p = (
        filename_end
        + extra_length
        + comment_length
    )

print("\nFIRST 10 ARCHIVE FILENAMES:")
for x in names[:10]:
    print(repr(x))

print("\nCASE-INSENSITIVE MATCH TEST:")

selected_lower = {
    x.lower()
    for x in selected_ids
}

matches = []

for name in names:

    stem = Path(name).stem

    video_id = stem.rsplit(
        "--",
        1
    )[0]

    if video_id.lower() in selected_lower:

        matches.append(
            (video_id, name)
        )

print(
    f"Matches among parsed entries: "
    f"{len(matches)}"
)

print("\nFIRST 10 MATCHES:")

for x in matches[:10]:
    print(x)