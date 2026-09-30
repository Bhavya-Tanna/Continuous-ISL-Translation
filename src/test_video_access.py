from huggingface_hub import HfFileSystem


REPO = "datasets/Exploration-Lab/iSign"

VIDEO_PART_A = "iSign-videos_v1.1_part_aa"
VIDEO_PART_B = "iSign-videos_v1.1_part_ab"


print("=" * 60)
print("iSIGN REMOTE ACCESS TEST")
print("=" * 60)


# ------------------------------------------------------------
# Connect to Hugging Face
# ------------------------------------------------------------

fs = HfFileSystem()

print("\nConnected to Hugging Face.")


# ------------------------------------------------------------
# Get metadata for the two video archive parts
# ------------------------------------------------------------

for filename in [VIDEO_PART_A, VIDEO_PART_B]:

    path = f"{REPO}/{filename}"

    print("\n" + "-" * 60)
    print(f"FILE: {filename}")
    print("-" * 60)

    info = fs.info(path)

    print(f"Size: {info.get('size', 'unknown'):,} bytes")
    print(f"Type: {info.get('type', 'unknown')}")


# ------------------------------------------------------------
# IMPORTANT:
# Do NOT read the files yet.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("METADATA TEST COMPLETE")
print("=" * 60)

print(
    "\nNo video archive bytes were intentionally read."
)

print(
    "If the file sizes appear correctly, "
    "we proceed to the range-access test."
)