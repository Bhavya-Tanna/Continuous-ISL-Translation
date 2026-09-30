import struct
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_VIDEOS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "selected_video_ids.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "selected_video_index.csv"
)

INDEX_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "zip_central_directory.bin"
)


# ============================================================
# KNOWN ZIP64 INFORMATION
# ============================================================

CENTRAL_DIRECTORY_OFFSET_IN_PART_B = 25_671_270_010
CENTRAL_DIRECTORY_SIZE = 15_002_335

# Physical size of the first video archive part.
# Only individual ZIP entries that fit completely inside part_aa are kept.
PART_AA_SIZE = 32_212_254_720


# ============================================================
# CONSTANTS
# ============================================================

CENTRAL_SIGNATURE = b"PK\x01\x02"

ZIP64_EXTRA_ID = 0x0001

ZIP32_MAX = 0xFFFFFFFF


print("=" * 70)
print("iSIGN SELECTIVE VIDEO INDEX - ZIP64 AWARE")
print("=" * 70)


# ============================================================
# LOAD CENTRAL DIRECTORY
# ============================================================

if INDEX_CACHE_PATH.exists():

    print(
        "\nUsing existing cached central directory:"
    )

    print(
        INDEX_CACHE_PATH
    )

    with open(
        INDEX_CACHE_PATH,
        "rb"
    ) as f:

        central_directory = f.read()

else:

    print(
        "\nCached central directory not found."
    )

    print(
        "The existing cache is required for this version."
    )

    raise FileNotFoundError(
        INDEX_CACHE_PATH
    )


print(
    f"\nCentral directory bytes: "
    f"{len(central_directory):,}"
)

if len(central_directory) != CENTRAL_DIRECTORY_SIZE:

    raise RuntimeError(
        "Central directory size mismatch."
    )


# ============================================================
# LOAD SELECTED VIDEO IDs
# ============================================================

selected = pd.read_csv(
    SELECTED_VIDEOS_PATH
)

selected_ids = set(
    selected["video_id"]
    .astype(str)
)

print(
    f"\nSelected video IDs: "
    f"{len(selected_ids):,}"
)


# ============================================================
# ZIP64 EXTRA FIELD PARSER
# ============================================================

def parse_zip64_extra(
    extra_data,
    compressed_size_32,
    uncompressed_size_32,
    local_header_offset_32,
    disk_number_32,
):
    """
    Parse ZIP64 extended information extra field.

    ZIP64 extra field ID:
        0x0001

    Values appear only when their corresponding
    normal ZIP32 field contains 0xFFFFFFFF.
    """

    position = 0

    zip64_uncompressed_size = None
    zip64_compressed_size = None
    zip64_local_header_offset = None
    zip64_disk_number = None

    while position + 4 <= len(extra_data):

        header_id, data_size = struct.unpack(
            "<HH",
            extra_data[position:position + 4]
        )

        data_start = position + 4
        data_end = data_start + data_size

        if data_end > len(extra_data):
            break

        field_data = extra_data[
            data_start:data_end
        ]

        # ----------------------------------------------------
        # ZIP64 Extended Information
        # ----------------------------------------------------

        if header_id == ZIP64_EXTRA_ID:

            field_position = 0

            # ----------------------------------------------
            # Uncompressed size
            # ----------------------------------------------

            if (
                uncompressed_size_32
                == ZIP32_MAX
            ):

                if field_position + 8 > len(field_data):
                    raise RuntimeError(
                        "Incomplete ZIP64 uncompressed size"
                    )

                zip64_uncompressed_size = struct.unpack(
                    "<Q",
                    field_data[
                        field_position:
                        field_position + 8
                    ]
                )[0]

                field_position += 8

            # ----------------------------------------------
            # Compressed size
            # ----------------------------------------------

            if (
                compressed_size_32
                == ZIP32_MAX
            ):

                if field_position + 8 > len(field_data):
                    raise RuntimeError(
                        "Incomplete ZIP64 compressed size"
                    )

                zip64_compressed_size = struct.unpack(
                    "<Q",
                    field_data[
                        field_position:
                        field_position + 8
                    ]
                )[0]

                field_position += 8

            # ----------------------------------------------
            # Local header offset
            # ----------------------------------------------

            if (
                local_header_offset_32
                == ZIP32_MAX
            ):

                if field_position + 8 > len(field_data):
                    raise RuntimeError(
                        "Incomplete ZIP64 local header offset"
                    )

                zip64_local_header_offset = struct.unpack(
                    "<Q",
                    field_data[
                        field_position:
                        field_position + 8
                    ]
                )[0]

                field_position += 8

            # ----------------------------------------------
            # Disk number
            # ----------------------------------------------

            if (
                disk_number_32
                == ZIP32_MAX
            ):

                if field_position + 4 > len(field_data):
                    raise RuntimeError(
                        "Incomplete ZIP64 disk number"
                    )

                zip64_disk_number = struct.unpack(
                    "<I",
                    field_data[
                        field_position:
                        field_position + 4
                    ]
                )[0]

            break

        position = data_end

    # ========================================================
    # Resolve final values
    # ========================================================

    final_uncompressed_size = (
        zip64_uncompressed_size
        if uncompressed_size_32 == ZIP32_MAX
        else uncompressed_size_32
    )

    final_compressed_size = (
        zip64_compressed_size
        if compressed_size_32 == ZIP32_MAX
        else compressed_size_32
    )

    final_local_header_offset = (
        zip64_local_header_offset
        if local_header_offset_32 == ZIP32_MAX
        else local_header_offset_32
    )

    final_disk_number = (
        zip64_disk_number
        if disk_number_32 == ZIP32_MAX
        else disk_number_32
    )

    # --------------------------------------------------------
    # Validate required ZIP64 values
    # --------------------------------------------------------

    if final_local_header_offset is None:

        raise RuntimeError(
            "ZIP64 local header offset was required "
            "but not found."
        )

    if final_compressed_size is None:

        raise RuntimeError(
            "ZIP64 compressed size was required "
            "but not found."
        )

    if final_uncompressed_size is None:

        raise RuntimeError(
            "ZIP64 uncompressed size was required "
            "but not found."
        )

    return (
        final_compressed_size,
        final_uncompressed_size,
        final_local_header_offset,
        final_disk_number,
    )


# ============================================================
# PARSE CENTRAL DIRECTORY
# ============================================================

position = 0

entries = []

zip64_entries = 0

central_length = len(
    central_directory
)

print(
    "\nParsing ZIP central directory..."
)


while position + 46 <= central_length:

    signature = (
        central_directory[
            position:
            position + 4
        ]
    )

    if signature != CENTRAL_SIGNATURE:

        break

    header = central_directory[
        position:
        position + 46
    ]

    (
        signature,
        version_made,
        version_needed,
        flags,
        compression,
        mod_time,
        mod_date,
        crc32,
        compressed_size_32,
        uncompressed_size_32,
        filename_length,
        extra_length,
        comment_length,
        disk_number_32,
        internal_attributes,
        external_attributes,
        local_header_offset_32,
    ) = struct.unpack(
        "<4s6H3I5H2I",
        header
    )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    filename_start = (
        position + 46
    )

    filename_end = (
        filename_start
        + filename_length
    )

    filename_bytes = (
        central_directory[
            filename_start:
            filename_end
        ]
    )

    try:

        filename = filename_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        filename = filename_bytes.decode(
            "utf-8",
            errors="replace"
        )

    # --------------------------------------------------------
    # Extra field
    # --------------------------------------------------------

    extra_start = filename_end

    extra_end = (
        extra_start
        + extra_length
    )

    extra_data = (
        central_directory[
            extra_start:
            extra_end
        ]
    )

    # --------------------------------------------------------
    # ZIP64 detection
    # --------------------------------------------------------

    needs_zip64 = (
        compressed_size_32 == ZIP32_MAX
        or
        uncompressed_size_32 == ZIP32_MAX
        or
        local_header_offset_32 == ZIP32_MAX
        or
        disk_number_32 == ZIP32_MAX
    )

    if needs_zip64:

        zip64_entries += 1

        (
            compressed_size,
            uncompressed_size,
            local_header_offset,
            disk_number,
        ) = parse_zip64_extra(
            extra_data,
            compressed_size_32,
            uncompressed_size_32,
            local_header_offset_32,
            disk_number_32,
        )

    else:

        compressed_size = (
            compressed_size_32
        )

        uncompressed_size = (
            uncompressed_size_32
        )

        local_header_offset = (
            local_header_offset_32
        )

        disk_number = (
            disk_number_32
        )

    # --------------------------------------------------------
    # Store entry
    # --------------------------------------------------------

    entries.append(
        {
            "filename": filename,
            "compression": compression,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_header_offset": local_header_offset,
            "disk_number": disk_number,
            "filename_length": filename_length,
            "extra_length": extra_length,
        }
    )

    # --------------------------------------------------------
    # Move to next central-directory entry
    # --------------------------------------------------------

    position = (
        extra_end
        + comment_length
    )


print(
    f"\nParsed entries: "
    f"{len(entries):,}"
)

print(
    f"ZIP64 entries: "
    f"{zip64_entries:,}"
)


# ============================================================
# CHECK FOR INVALID 32-BIT OFFSETS
# ============================================================

invalid_offsets = [
    entry
    for entry in entries
    if entry["local_header_offset"]
    == ZIP32_MAX
]

print(
    f"Entries still using "
    f"0xFFFFFFFF offset: "
    f"{len(invalid_offsets):,}"
)

if len(invalid_offsets) > 0:

    raise RuntimeError(
        "Some entries still have unresolved "
        "ZIP64 offsets."
    )


# ============================================================
# SEARCH SELECTED VIDEO IDs
# ============================================================

print(
    "\nSearching for selected video IDs..."
)

selected_entries = []

for entry in entries:

    filename = entry["filename"]

    matched_video_id = None

    for video_id in selected_ids:

        # Video IDs may themselves contain or end with '-'
        # so use substring matching rather than splitting
        # on '--'.

        if video_id in filename:

            matched_video_id = video_id
            break

    if matched_video_id is not None:

        # --------------------------------------------------------
        # PART-AA-ONLY FILTER
        # --------------------------------------------------------
        # Keep this individual ZIP entry only if its complete
        # local header + filename + extra field + compressed data
        # fits inside part_aa.
        #
        # Entries that start in part_ab, or cross the boundary
        # from part_aa into part_ab, are skipped.
        entry_end = (
            entry["local_header_offset"]
            + 30
            + entry["filename_length"]
            + entry["extra_length"]
            + entry["compressed_size"]
        )

        if (
            entry["local_header_offset"] >= PART_AA_SIZE
            or entry_end > PART_AA_SIZE
        ):
            continue

        row = entry.copy()

        row["video_id"] = (
            matched_video_id
        )

        selected_entries.append(
            row
        )

# ============================================================
# RESULTS
# ============================================================

matched_df = pd.DataFrame(
    selected_entries
)

print(
    "\n" + "-" * 70
)

print(
    "MATCHING RESULTS"
)

print(
    "-" * 70
)

print(
    f"Selected video IDs       : "
    f"{len(selected_ids):,}"
)

print(
    f"Matched archive entries  : "
    f"{len(matched_df):,}"
)

print(
    "Part-ab / cross-boundary entries dropped: "
    "individual ZIP entries that do not fit completely in part_aa"
)


if len(matched_df) > 0:

    matched_video_ids = set(
        matched_df["video_id"]
    )

    print(
        f"Unique matched video IDs : "
        f"{len(matched_video_ids):,}"
    )

    missing_ids = (
        selected_ids
        - matched_video_ids
    )

    print(
        f"Missing video IDs        : "
        f"{len(missing_ids):,}"
    )

    # --------------------------------------------------------
    # Check unresolved offsets
    # --------------------------------------------------------

    bad_selected_offsets = matched_df[
        matched_df["local_header_offset"]
        == ZIP32_MAX
    ]

    print(
        f"Selected entries with "
        f"invalid offset            : "
        f"{len(bad_selected_offsets):,}"
    )

    # --------------------------------------------------------
    # Show the problematic entry
    # --------------------------------------------------------

    problem_entry = matched_df[
        matched_df["filename"].str.contains(
            "W_LjNlwlqSI--10.mp4",
            regex=False
        )
    ]

    if len(problem_entry) > 0:

        print(
            "\nVerification: "
            "W_LjNlwlqSI--10.mp4"
        )

        print(
            problem_entry[
                [
                    "filename",
                    "compressed_size",
                    "uncompressed_size",
                    "local_header_offset",
                    "disk_number",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # First entries
    # --------------------------------------------------------

    print(
        "\nFirst 10 selected entries:"
    )

    print(
        matched_df.head(10).to_string(
            index=False
        )
    )


else:

    print(
        "\nNo selected video IDs were matched."
    )


# ============================================================
# SAVE INDEX
# ============================================================

if len(matched_df) > 0:

    matched_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved corrected index to:"
        f"\n{OUTPUT_PATH}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(matched_df) > 0:

    print(
        "\nRunning final validation..."
    )

    # All offsets should be integers
    if (
        matched_df[
            "local_header_offset"
        ]
        < 0
    ).any():

        raise RuntimeError(
            "Negative local header offset detected."
        )

    # No unresolved ZIP32 sentinel
    if (
        matched_df[
            "local_header_offset"
        ]
        == ZIP32_MAX
    ).any():

        raise RuntimeError(
            "Unresolved 0xFFFFFFFF offset remains."
        )

    # No zero offsets
    if (
        matched_df[
            "local_header_offset"
        ]
        == 0
    ).any():

        raise RuntimeError(
            "Zero local header offset detected."
        )

    # Every retained individual entry must fit completely in part_aa.
    entry_end = (
        matched_df["local_header_offset"]
        + 30
        + matched_df["filename_length"]
        + matched_df["extra_length"]
        + matched_df["compressed_size"]
    )

    if (
        entry_end > PART_AA_SIZE
    ).any():

        raise RuntimeError(
            "An entry crosses the end of part_aa."
        )

    if (
        matched_df["local_header_offset"] >= PART_AA_SIZE
    ).any():

        raise RuntimeError(
            "An entry belongs to part_ab."
        )

    print(
        "✓ All selected offsets are valid."
    )

    print(
        "✓ ZIP64 offsets resolved."
    )

    print(
        "✓ All selected entries fit completely inside part_aa."
    )


print(
    "\n" + "=" * 70
)

print(
    "ZIP64-AWARE INDEX BUILD COMPLETE"
)

print(
    "=" * 70
)
