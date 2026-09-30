import time
import zlib
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_url, get_token


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "Exploration-Lab/iSign"
REPO_TYPE = "dataset"
REMOTE_FILE = "iSign-videos_v1.1_part_aa"

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
    / "videos"
)

FAILURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "download_failures.csv"
)

PROGRESS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "download_progress.csv"
)

MAX_RETRIES = 5
CHUNK_SIZE = 1024 * 1024
TIMEOUT = (30, 180)

# Retry delays for temporary HTTP/network failures.
RETRY_DELAYS = [5, 10, 20, 40, 60]

# Only entries from the corrected index are allowed.
PART_AA_SIZE = 32_212_254_720


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("iSIGN SELECTIVE VIDEO DOWNLOADER - DIRECT HTTP RANGE")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INDEX_PATH)

required_columns = {
    "filename",
    "compression",
    "compressed_size",
    "uncompressed_size",
    "local_header_offset",
    "filename_length",
    "extra_length",
    "video_id",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise RuntimeError(
        f"selected_video_index.csv is missing columns: "
        f"{sorted(missing_columns)}"
    )

print(f"\nIndex entries: {len(df):,}")


# ============================================================
# VALIDATE INDEX BEFORE DOWNLOADING
# ============================================================

df["local_header_offset"] = df["local_header_offset"].astype("int64")
df["compressed_size"] = df["compressed_size"].astype("int64")
df["uncompressed_size"] = df["uncompressed_size"].astype("int64")
df["filename_length"] = df["filename_length"].astype("int64")
df["extra_length"] = df["extra_length"].astype("int64")

entry_end = (
    df["local_header_offset"]
    + 30
    + df["filename_length"]
    + df["extra_length"]
    + df["compressed_size"]
)

if (df["local_header_offset"] < 0).any():
    raise RuntimeError("Negative ZIP offset found.")

if (df["local_header_offset"] >= PART_AA_SIZE).any():
    raise RuntimeError(
        "Index contains an entry outside part_aa."
    )

if (entry_end > PART_AA_SIZE).any():
    raise RuntimeError(
        "Index contains an entry crossing the end of part_aa."
    )

if (df["compression"] != 8).all() is False:
    bad = df[df["compression"] != 8]
    raise RuntimeError(
        f"Found {len(bad)} entries that are not DEFLATE (compression=8)."
    )

print("✓ Index validation passed.")
print("✓ All entries fit completely inside part_aa.")
print("✓ All entries use DEFLATE compression.")


# ============================================================
# HF URL + AUTH
# ============================================================

URL = hf_hub_url(
    repo_id=REPO_ID,
    filename=REMOTE_FILE,
    repo_type=REPO_TYPE,
)

TOKEN = get_token()

BASE_HEADERS = {}

if TOKEN:
    BASE_HEADERS["Authorization"] = f"Bearer {TOKEN}"

print(f"\nRemote archive part:")
print(REMOTE_FILE)


# ============================================================
# HTTP RANGE SESSION
# ============================================================

session = requests.Session()


def range_get(start, end):
    """
    Download exactly the requested byte range from part_aa.
    Returns response.content after validating the HTTP response.
    """

    headers = dict(BASE_HEADERS)
    headers["Range"] = f"bytes={start}-{end}"

    expected = end - start + 1

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                URL,
                headers=headers,
                allow_redirects=True,
                timeout=TIMEOUT,
            )

            status = response.status_code

            # Range request must be honored.
            if status == 206:

                content_range = response.headers.get(
                    "Content-Range",
                    ""
                )

                if len(response.content) != expected:
                    raise RuntimeError(
                        f"Range length mismatch: "
                        f"expected {expected}, "
                        f"received {len(response.content)}"
                    )

                # Basic Content-Range validation.
                expected_prefix = f"bytes {start}-{end}/"

                if not content_range.startswith(
                    expected_prefix
                ):
                    raise RuntimeError(
                        f"Unexpected Content-Range: "
                        f"{content_range}"
                    )

                return response.content

            # A server returning 200 means it ignored our Range header.
            if status == 200:
                raise RuntimeError(
                    "Server returned HTTP 200 instead of 206; "
                    "Range request was not honored."
                )

            # Temporary errors.
            if status in (408, 429, 500, 502, 503, 504):

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = RETRY_DELAYS[
                            min(attempt - 1, len(RETRY_DELAYS) - 1)
                        ]
                else:
                    delay = RETRY_DELAYS[
                        min(attempt - 1, len(RETRY_DELAYS) - 1)
                    ]

                last_error = RuntimeError(
                    f"HTTP {status}"
                )

                print(
                    f"    HTTP {status}; "
                    f"retrying in {delay:g}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )

                time.sleep(delay)
                continue

            raise RuntimeError(
                f"HTTP {status}: "
                f"{response.text[:200]}"
            )

        except (
            requests.RequestException,
            RuntimeError,
        ) as exc:

            last_error = exc

            if attempt >= MAX_RETRIES:
                break

            delay = RETRY_DELAYS[
                min(attempt - 1, len(RETRY_DELAYS) - 1)
            ]

            print(
                f"    Request error: {exc}; "
                f"retrying in {delay:g}s "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Range request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# DOWNLOAD ONE VIDEO
# ============================================================

def download_one(row):

    filename = str(row["filename"])
    output_name = Path(filename).name

    compressed_size = int(row["compressed_size"])
    expected_uncompressed = int(row["uncompressed_size"])
    local_header_offset = int(row["local_header_offset"])
    filename_length = int(row["filename_length"])
    extra_length = int(row["extra_length"])

    # Local ZIP header is 30 bytes.
    data_offset = (
        local_header_offset
        + 30
        + filename_length
        + extra_length
    )

    data_end = data_offset + compressed_size - 1

    # Safety check.
    if data_end >= PART_AA_SIZE:
        raise RuntimeError(
            "Compressed data extends beyond part_aa."
        )

    output_path = OUTPUT_DIR / output_name
    temp_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    # --------------------------------------------------------
    # Resume / existing-file validation
    # --------------------------------------------------------

    if output_path.exists():

        try:
            actual_size = output_path.stat().st_size

            if actual_size == expected_uncompressed:
                return "existing"

            print(
                f"    Existing file has wrong size "
                f"({actual_size} != {expected_uncompressed}); "
                f"redownloading."
            )

        except OSError:
            pass

    # Remove stale partial file.
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass

    # --------------------------------------------------------
    # One direct HTTP Range request for compressed data
    # --------------------------------------------------------

    compressed_data = range_get(
        data_offset,
        data_end,
    )

    # --------------------------------------------------------
    # DEFLATE decompression
    # --------------------------------------------------------

    try:
        decompressor = zlib.decompressobj(-15)

        uncompressed_data = (
            decompressor.decompress(compressed_data)
            + decompressor.flush()
        )

    except zlib.error as exc:

        raise RuntimeError(
            f"DEFLATE decompression failed: {exc}"
        )

    if len(uncompressed_data) != expected_uncompressed:

        raise RuntimeError(
            f"Uncompressed size mismatch: "
            f"expected {expected_uncompressed}, "
            f"received {len(uncompressed_data)}"
        )

    # --------------------------------------------------------
    # Atomic save
    # --------------------------------------------------------

    with open(temp_path, "wb") as f:
        f.write(uncompressed_data)

    temp_path.replace(output_path)

    return "downloaded"


# ============================================================
# DOWNLOAD LOOP
# ============================================================

failures = []
downloaded = 0
existing = 0
failed = 0

total = len(df)


print(
    f"\nStarting download of {total:,} entries..."
)

for position, (_, row) in enumerate(
    df.iterrows(),
    start=1,
):

    filename = str(row["filename"])
    output_name = Path(filename).name

    print(
        f"\n[{position}/{total}] {output_name}"
    )

    try:

        result = download_one(row)

        if result == "existing":
            existing += 1
            print("    ✓ Existing valid file")

        else:
            downloaded += 1
            print("    ✓ Downloaded")

    except Exception as exc:

        failed += 1

        print(
            f"    ✗ FAILED: {exc}"
        )

        failures.append(
            {
                "filename": filename,
                "video_id": row["video_id"],
                "error": str(exc),
            }
        )

    if position % 25 == 0 or position == total:

        print(
            "\nProgress:"
            f" downloaded={downloaded:,}"
            f" existing={existing:,}"
            f" failed={failed:,}"
        )


# ============================================================
# SAVE FAILURE REPORT
# ============================================================

if failures:

    pd.DataFrame(failures).to_csv(
        FAILURE_PATH,
        index=False,
    )

    print(
        f"\nFailure report saved to:"
        f"\n{FAILURE_PATH}"
    )

else:

    # Remove stale failure report if everything succeeded.
    if FAILURE_PATH.exists():
        FAILURE_PATH.unlink()

    print(
        "\n✓ No download failures."
    )


# ============================================================
# SAVE PROGRESS
# ============================================================

progress = pd.DataFrame(
    [
        {
            "total_entries": total,
            "downloaded": downloaded,
            "existing": existing,
            "failed": failed,
        }
    ]
)

progress.to_csv(
    PROGRESS_PATH,
    index=False,
)

print(
    f"\nProgress saved to:"
    f"\n{PROGRESS_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"Total entries : {total:,}")
print(f"Downloaded    : {downloaded:,}")
print(f"Already valid : {existing:,}")
print(f"Failed        : {failed:,}")

print("=" * 70)
