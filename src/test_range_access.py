from huggingface_hub import HfFileSystem


REPO = "datasets/Exploration-Lab/iSign"
FILE = "iSign-videos_v1.1_part_ab"

REMOTE_PATH = f"{REPO}/{FILE}"

# Only 64 KB
READ_SIZE = 64 * 1024


print("=" * 60)
print("iSIGN RANGE ACCESS TEST")
print("=" * 60)

fs = HfFileSystem()

# ------------------------------------------------------------
# Get remote file size
# ------------------------------------------------------------

info = fs.info(REMOTE_PATH)

file_size = info["size"]

print(f"\nRemote file:")
print(f"  {FILE}")

print(f"\nRemote size:")
print(f"  {file_size:,} bytes")
print(
    f"  {file_size / (1024 ** 3):.2f} GB"
)

# ------------------------------------------------------------
# Read ONLY the final 64 KB
# ------------------------------------------------------------

start_position = max(0, file_size - READ_SIZE)

print("\nRequesting only:")
print(f"  bytes {start_position:,} → {file_size:,}")

print("\nOpening remote file...")

with fs.open(REMOTE_PATH, "rb") as f:

    f.seek(start_position)

    data = f.read(READ_SIZE)

print("\nRange read completed.")

print(f"Bytes received: {len(data):,}")

# ------------------------------------------------------------
# Inspect ZIP signatures
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("ZIP SIGNATURE CHECK")
print("-" * 60)

signatures = {
    b"PK\x03\x04": "ZIP local file header",
    b"PK\x01\x02": "ZIP central directory entry",
    b"PK\x05\x06": "ZIP end of central directory",
    b"PK\x06\x06": "ZIP64 end of central directory",
    b"PK\x06\x07": "ZIP64 end of central directory locator",
}

found = False

for signature, description in signatures.items():

    position = data.find(signature)

    if position != -1:

        print(
            f"FOUND: {description} "
            f"at range offset {position:,}"
        )

        found = True

if not found:
    print("No ZIP signature found in final 64 KB.")

# ------------------------------------------------------------
# Show first/last bytes
# ------------------------------------------------------------

print("\nFirst 32 bytes:")
print(data[:32].hex(" "))

print("\nLast 32 bytes:")
print(data[-32:].hex(" "))

print("\n" + "=" * 60)
print("RANGE ACCESS TEST COMPLETE")
print("=" * 60)