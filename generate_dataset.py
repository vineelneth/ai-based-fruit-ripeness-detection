"""
Flatten the raw dataset directory into a single image folder and generate
a regression labels CSV (ripeness_pct, days_to_ripe) derived from folder names.
"""

import argparse
import csv
import os
import random
import shutil


def generate_labels(folder_name: str) -> tuple[float, float]:
    """Return (ripeness_pct, days_to_ripe) based on the source folder name."""
    name = folder_name.lower()
    if "unripe" in name:
        return round(random.uniform(10.0, 45.0), 1), round(random.uniform(5.0, 10.0), 1)
    elif "fresh" in name:
        return round(random.uniform(70.0, 95.0), 1), round(random.uniform(0.5, 3.0), 1)
    elif "rotten" in name:
        return 100.0, 0.0
    else:
        return round(random.uniform(40.0, 60.0), 1), round(random.uniform(4.0, 7.0), 1)


def build_dataset(source_base: str, dest_dir: str, csv_file: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    csv_rows = [["filename", "ripeness_pct", "days_to_ripe"]]

    print(f"Scanning {source_base}/ and copying images to {dest_dir}/ ...")
    for split in ("train", "test"):
        split_path = os.path.join(source_base, split)
        if not os.path.exists(split_path):
            print(f"  Warning: {split_path} not found, skipping.")
            continue

        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            safe_class = class_folder.replace(" ", "_")
            ripeness, days = generate_labels(class_folder)

            for fname in os.listdir(class_path):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                new_name = f"{split}_{safe_class}_{fname}"
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(dest_dir, new_name),
                )
                csv_rows.append([new_name, ripeness, days])

    with open(csv_file, mode="w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nDone.")
    print(f"  Images copied to : {dest_dir}")
    print(f"  Labels written to: {csv_file}  ({len(csv_rows) - 1} rows)")


def main():
    parser = argparse.ArgumentParser(description="Generate flattened dataset and labels CSV.")
    parser.add_argument("--source", default="dataset", help="Root dataset directory")
    parser.add_argument("--dest-dir", default="dataset/all_images", help="Destination image folder")
    parser.add_argument("--csv-out", default="fruit_labels.csv", help="Output labels CSV path")
    args = parser.parse_args()

    build_dataset(args.source, args.dest_dir, args.csv_out)


if __name__ == "__main__":
    main()
