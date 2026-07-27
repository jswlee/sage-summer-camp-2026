import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

CLASS_NAMES = ("good", "bad")
SPLIT_NAMES = ("train", "val", "test")


def parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("America/Chicago"))


def build_image_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root, _dirs, files in os.walk(images_dir):
        for name in files:
            if name.lower().endswith(".jpg"):
                index[name] = Path(root) / name
    return index


def classify(pm25: float, threshold: float) -> str:
    return "bad" if pm25 >= threshold else "good"


def split_indices(n: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    return train_end, val_end


def copy_image(src: Path, dst: Path, imgsz: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if imgsz <= 0:
        shutil.copy2(src, dst)
        return
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize((imgsz, imgsz), Image.LANCZOS)
        img.save(dst, "JPEG", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Sage images into a YOLO classification dataset "
        "(train/val/test) with good/bad air labels."
    )
    parser.add_argument("--csv", type=Path, default=Path("all_data.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: yolo_dataset_<time_of_day>.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=55.5,
        help="purple_air_pm25 below this is 'good', at/above is 'bad'.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Output image size (square). Use 0 to copy images unchanged.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.7, 0.2, 0.1),
        help="Train/val/test split ratios (must sum to 1.0).",
    )
    parser.add_argument(
        "--time-of-day",
        type=str,
        choices=["both", "day", "night"],
        default="both",
        help="Time filter in Chicago time: 'day' = 05:00-21:00, "
        "'night' = all other hours, 'both' = all hours.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"yolo_dataset_{args.time_of_day}")

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        parser.error(f"--ratios must sum to 1.0, got {sum(args.ratios)}")

    random.seed(args.seed)
    index = build_image_index(args.images_dir)
    if not index:
        parser.error(f"No .jpg images found under {args.images_dir}")

    # Group images by (date, label) so the split is stratified per day.
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    missing_image = 0
    missing_pm25 = 0

    with args.csv.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            filename = row["filename"].strip()
            pm25_raw = row["purple_air_pm25"].strip()
            timestamp = parse_timestamp(row["timestamp"])
            date = timestamp.date().isoformat()
            hour = timestamp.hour

            if args.time_of_day == "day" and not (5 <= hour < 21):
                continue
            if args.time_of_day == "night" and (5 <= hour < 21):
                continue

            if not pm25_raw:
                missing_pm25 += 1
                continue
            if filename not in index:
                missing_image += 1
                continue

            label = classify(float(pm25_raw), args.threshold)
            groups[(date, label)].append(filename)

    # Assign each group's images to splits, then copy.
    counts: dict[str, dict[str, int]] = {
        split: {cls: 0 for cls in CLASS_NAMES} for split in SPLIT_NAMES
    }
    total_copied = 0

    for (date, label), filenames in sorted(groups.items()):
        random.shuffle(filenames)
        train_end, val_end = split_indices(len(filenames), tuple(args.ratios))
        assignments = {
            "train": filenames[:train_end],
            "val": filenames[train_end:val_end],
            "test": filenames[val_end:],
        }
        for split, split_files in assignments.items():
            for filename in split_files:
                dst = args.output / split / label / filename
                copy_image(index[filename], dst, args.imgsz)
                counts[split][label] += 1
                total_copied += 1

    # Balance each split by removing random surplus images from the majority class.
    removed = 0
    for split in SPLIT_NAMES:
        split_dir = args.output / split
        good_files = list((split_dir / "good").glob("*.jpg"))
        bad_files = list((split_dir / "bad").glob("*.jpg"))
        target = min(len(good_files), len(bad_files))
        for files in (good_files, bad_files):
            if len(files) > target:
                random.shuffle(files)
                for f in files[target:]:
                    f.unlink()
                    removed += 1
        counts[split]["good"] = target
        counts[split]["bad"] = target

    total_copied -= removed

    print(f"Copied {total_copied} images into {args.output}")
    if removed:
        print(f"  Removed {removed} surplus images to balance good/bad per split")
    if missing_image:
        print(f"  Skipped {missing_image} rows with no matching image file")
    if missing_pm25:
        print(f"  Skipped {missing_pm25} rows with empty purple_air_pm25")
    print(f"  Time of day: {args.time_of_day} (Chicago time)")
    print(f"  Threshold: pm2.5 >= {args.threshold} -> bad, else good")
    print("  Split breakdown:")
    header = f"    {'split':<6} {'good':>7} {'bad':>7} {'total':>7}"
    print(header)
    for split in SPLIT_NAMES:
        good = counts[split]["good"]
        bad = counts[split]["bad"]
        print(f"    {split:<6} {good:>7} {bad:>7} {good + bad:>7}")


if __name__ == "__main__":
    main()
