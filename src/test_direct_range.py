import requests
import pandas as pd
from pathlib import Path

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

# ============================================================
# LOAD INDEX
# ============================================================

df = pd.read_csv(INDEX_PATH)

print("Columns:", list(df.columns))
print(df.head(3).to_string())
print("Total rows:", len(df))


# ============================================================
# SELECT TARGET VIDEO
# ============================================================

TARGET_ID = "rIAx3FPvJhg"
TARGET_SUFFIX = "--9.mp4"

target = df[
    df["filename"].astype(str).str.contains(
        TARGET_ID,
        case=False,
        regex=False
    )
    &
    df["filename"].astype(str).str.endswith(
        TARGET_SUFFIX
    )
]

print("\nSelected target:")
print(target.to_string(index=False))

if len(target) != 1:
    raise RuntimeError(
        f"Expected exactly one target, found {len(target)}"
    )

row = target.iloc[0]

offset = int(row["local_header_offset"])
compressed_size = int(row["compressed_size"])

print("=" * 70)
print("DIRECT HTTP RANGE TEST")
print("=" * 70)

print(
    f"\nFile:"
    f"\n{row['filename']}"
)

print(
    f"\nOffset:"
    f"\n{offset:,}"
)

print(
    f"\nCompressed size:"
    f"\n{compressed_size:,}"
)

# ============================================================
# BUILD HF RESOLVE URL
# ============================================================

url = hf_hub_url(
    repo_id=REPO_ID,
    filename=REMOTE_FILE,
    repo_type=REPO_TYPE,
)

print(
    f"\nResolve URL:"
    f"\n{url}"
)


# ============================================================
# AUTH
# ============================================================

token = get_token()

headers = {
    "Range": f"bytes={offset}-{offset + 29}",
}

if token:

    headers["Authorization"] = (
        f"Bearer {token}"
    )


# ============================================================
# DIRECT HTTP RANGE REQUEST
# ============================================================

print(
    "\nRequesting exactly 30 bytes..."
)

response = requests.get(
    url,
    headers=headers,
    allow_redirects=True,
    timeout=(30, 120),
)

print(
    f"\nHTTP status: "
    f"{response.status_code}"
)

print(
    f"Received bytes: "
    f"{len(response.content)}"
)

print(
    f"Content-Range: "
    f"{response.headers.get('Content-Range')}"
)

print(
    f"Content-Length: "
    f"{response.headers.get('Content-Length')}"
)

print(
    "\nBytes:"
)

print(
    response.content.hex(" ")
)

print(
    "\nFirst 4 bytes:"
)

print(
    response.content[:4]
)


# ============================================================
# VALIDATION
# ============================================================

if response.status_code not in (200, 206):

    raise RuntimeError(
        f"HTTP request failed: "
        f"{response.status_code}"
    )

if len(response.content) < 30:

    raise RuntimeError(
        f"Expected at least 30 bytes, "
        f"received {len(response.content)}"
    )

if response.content[:4] != b"PK\x03\x04":

    raise RuntimeError(
        "Received bytes are NOT a ZIP local header."
    )


print(
    "\n" + "=" * 70
)

print(
    "DIRECT RANGE TEST SUCCESSFUL"
)

print(
    "=" * 70
)

print(
    "\nThe HTTP Range method correctly reached "
    "the ZIP local header."
)