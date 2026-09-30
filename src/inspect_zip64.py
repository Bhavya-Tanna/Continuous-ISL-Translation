import struct
from huggingface_hub import HfFileSystem


REPO = "datasets/Exploration-Lab/iSign"
FILE = "iSign-videos_v1.1_part_ab"

REMOTE_PATH = f"{REPO}/{FILE}"

READ_SIZE = 1024 * 1024  # 1 MB


print("=" * 60)
print("iSIGN ZIP64 DIRECTORY INSPECTION")
print("=" * 60)


fs = HfFileSystem()

info = fs.info(REMOTE_PATH)
file_size = info["size"]

print(f"\nRemote file size: {file_size:,} bytes")
print(f"Remote file size: {file_size / (1024 ** 3):.2f} GB")


# ------------------------------------------------------------
# Read only final 1 MB
# ------------------------------------------------------------

start = file_size - READ_SIZE

print(
    f"\nRequesting only final "
    f"{READ_SIZE:,} bytes..."
)

with fs.open(REMOTE_PATH, "rb") as f:
    f.seek(start)
    data = f.read(READ_SIZE)

print(f"Received: {len(data):,} bytes")


# ------------------------------------------------------------
# Locate ZIP64 End of Central Directory
# ------------------------------------------------------------

ZIP64_EOCD = b"PK\x06\x06"
ZIP64_LOCATOR = b"PK\x06\x07"
EOCD = b"PK\x05\x06"


zip64_pos = data.rfind(ZIP64_EOCD)
locator_pos = data.rfind(ZIP64_LOCATOR)
eocd_pos = data.rfind(EOCD)


print("\n" + "-" * 60)
print("ZIP RECORD LOCATIONS")
print("-" * 60)

print(
    f"ZIP64 EOCD position in buffer : {zip64_pos:,}"
)

print(
    f"ZIP64 locator position        : {locator_pos:,}"
)

print(
    f"Normal EOCD position          : {eocd_pos:,}"
)


if zip64_pos == -1:
    raise RuntimeError(
        "ZIP64 End of Central Directory not found."
    )


# ------------------------------------------------------------
# Parse ZIP64 End of Central Directory
# ------------------------------------------------------------

# Structure:
#
# 4 bytes  signature
# 8 bytes  size of record
# 2 bytes  version made by
# 2 bytes  version needed
# 4 bytes  disk number
# 4 bytes  disk with central directory
# 8 bytes  entries on disk
# 8 bytes  total entries
# 8 bytes  central directory size
# 8 bytes  central directory offset

header = data[zip64_pos:zip64_pos + 56]

(
    signature,
    record_size,
    version_made,
    version_needed,
    disk_number,
    cd_disk,
    entries_disk,
    total_entries,
    cd_size,
    cd_offset
) = struct.unpack(
    "<4sQHHIIQQQQ",
    header
)


# ------------------------------------------------------------
# Convert relative archive offset
# ------------------------------------------------------------

PART_A_SIZE = 32_212_254_720

print("\n" + "-" * 60)
print("ZIP64 INFORMATION")
print("-" * 60)

print(f"Record size              : {record_size:,}")
print(f"Version needed           : {version_needed}")
print(f"Total entries            : {total_entries:,}")
print(f"Central directory size   : {cd_size:,}")
print(f"Central directory offset : {cd_offset:,}")


# ------------------------------------------------------------
# Important: archive is split into two physical files
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("PART-B OFFSET")
print("-" * 60)

if cd_offset >= PART_A_SIZE:

    part_b_offset = cd_offset - PART_A_SIZE

    print(
        "Central directory is inside part_ab."
    )

    print(
        f"Offset inside part_ab : "
        f"{part_b_offset:,}"
    )

else:

    print(
        "Central directory begins inside part_aa."
    )

    print(
        "We will need to inspect the split boundary."
    )


print("\n" + "=" * 60)
print("ZIP64 INSPECTION COMPLETE")
print("=" * 60)