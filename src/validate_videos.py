from pathlib import Path
import cv2
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "data" / "metadata" / "selected_video_index.csv"
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "video_integrity.csv"


# ============================================================
# VIDEO VALIDATION
# ============================================================

def validate_video(row):
    filename = row["filename"]

    # The downloader stores files using only the basename.
    video_path = VIDEO_DIR / Path(filename).name

    result = {
        "filename": filename,
        "video_id": row["video_id"],
        "expected_size": int(row["uncompressed_size"]),
        "actual_size": None,
        "exists": False,
        "size_valid": False,
        "readable": False,
        "first_frame_valid": False,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
        "status": None,
        "error": None,
    }

    # --------------------------------------------------------
    # Existence
    # --------------------------------------------------------

    if not video_path.exists():
        result["status"] = "missing"
        result["error"] = "File does not exist"
        return result

    result["exists"] = True

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    actual_size = video_path.stat().st_size
    result["actual_size"] = actual_size
    result["size_valid"] = (
        actual_size == result["expected_size"]
    )

    if not result["size_valid"]:
        result["status"] = "size_mismatch"
        result["error"] = (
            f"Expected {result['expected_size']} bytes, "
            f"found {actual_size} bytes"
        )
        return result

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        result["status"] = "unreadable"
        result["error"] = "OpenCV could not open the video"
        cap.release()
        return result

    result["readable"] = True

    # --------------------------------------------------------
    # Video properties
    # --------------------------------------------------------

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result["width"] = width
    result["height"] = height
    result["fps"] = fps
    result["frame_count"] = frame_count

    if fps > 0 and frame_count > 0:
        result["duration_seconds"] = frame_count / fps

    # --------------------------------------------------------
    # Decode first frame
    # --------------------------------------------------------

    success, frame = cap.read()

    if success and frame is not None:
        result["first_frame_valid"] = True
    else:
        result["status"] = "frame_decode_failed"
        result["error"] = "Could not decode first frame"
        cap.release()
        return result

    cap.release()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if width <= 0 or height <= 0:
        result["status"] = "invalid_resolution"
        result["error"] = "Invalid video resolution"
        return result

    if fps <= 0:
        result["status"] = "invalid_fps"
        result["error"] = "Invalid FPS"
        return result

    if frame_count <= 0:
        result["status"] = "invalid_frame_count"
        result["error"] = "Invalid frame count"
        return result

    result["status"] = "valid"

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VIDEO INTEGRITY VALIDATION")
    print("=" * 60)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found:\n{INDEX_PATH}"
        )

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory not found:\n{VIDEO_DIR}"
        )

    df = pd.read_csv(INDEX_PATH)

    print(f"\nIndexed entries: {len(df):,}")
    print(f"Video directory: {VIDEO_DIR}")

    # --------------------------------------------------------
    # Duplicate filename check
    # --------------------------------------------------------

    duplicate_count = df["filename"].duplicated().sum()

    print(f"Duplicate index filenames: {duplicate_count}")

    if duplicate_count > 0:
        print("WARNING: Duplicate filenames detected.")

    # --------------------------------------------------------
    # Validate every video
    # --------------------------------------------------------

    results = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        result = validate_video(row)
        results.append(result)

        if i % 100 == 0 or i == len(df):
            print(
                f"Validated {i:,}/{len(df):,} videos"
            )

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(result_df)

    valid = (
        result_df["status"] == "valid"
    ).sum()

    missing = (
        result_df["status"] == "missing"
    ).sum()

    size_mismatch = (
        result_df["status"] == "size_mismatch"
    ).sum()

    unreadable = (
        result_df["status"] == "unreadable"
    ).sum()

    frame_decode_failed = (
        result_df["status"] == "frame_decode_failed"
    ).sum()

    invalid_resolution = (
        result_df["status"] == "invalid_resolution"
    ).sum()

    invalid_fps = (
        result_df["status"] == "invalid_fps"
    ).sum()

    invalid_frame_count = (
        result_df["status"] == "invalid_frame_count"
    ).sum()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total videos:          {total:,}")
    print(f"Valid videos:          {valid:,}")
    print(f"Missing:               {missing:,}")
    print(f"Size mismatch:         {size_mismatch:,}")
    print(f"Unreadable:            {unreadable:,}")
    print(f"Frame decode failure:  {frame_decode_failed:,}")
    print(f"Invalid resolution:    {invalid_resolution:,}")
    print(f"Invalid FPS:           {invalid_fps:,}")
    print(f"Invalid frame count:   {invalid_frame_count:,}")

    print("\nReport saved to:")
    print(OUTPUT_PATH)

    if valid == total:
        print("\n✓ ALL VIDEOS PASSED INTEGRITY VALIDATION")
    else:
        print(
            f"\n⚠ {total - valid:,} VIDEOS REQUIRE ATTENTION"
        )


if __name__ == "__main__":
    main()