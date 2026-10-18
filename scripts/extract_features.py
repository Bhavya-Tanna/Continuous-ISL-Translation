"""Extract cached ResNet-18 features for every unique video in the manifest."""
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.dataset_loader import _resolve_video_path, _split_filenames
from src.feature_extractor import ResNet18FrameExtractor, extract_to_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/metadata/dataset_splits.csv")
    p.add_argument("--video-dir", required=True, help="Folder containing the real videos")
    p.add_argument("--cache-dir", default="data/processed/resnet18_features")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    df = pd.read_csv(args.manifest)
    paths, seen = [], set()
    for cell in df["video_filenames"]:
        for name in _split_filenames(cell):
            path = _resolve_video_path(name, Path(args.video_dir))
            key = str(path.resolve())
            if key not in seen:
                paths.append(path); seen.add(key)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} videos missing. First few:\n" + "\n".join(map(str, missing[:10])))
    extractor = ResNet18FrameExtractor(args.device, args.frames, not args.no_pretrained)
    print(f"Unique video files: {len(paths)}")
    for path in tqdm(paths, desc="Extracting"):
        extract_to_cache(path, args.cache_dir, extractor, args.overwrite)
    print("Cache:", Path(args.cache_dir).resolve())


if __name__ == "__main__":
    main()
