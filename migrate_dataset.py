#!/usr/bin/env python
"""
Migrate dataset/all_images/ + fruit_labels.csv to the per-fruit folder structure.

Before:
    dataset/all_images/train_freshapples_*.png   (564 files, flat)
    fruit_labels.csv                              (19 956 rows incl. augmented)

After:
    dataset/apple/   + labels.csv
    dataset/banana/  + labels.csv
    dataset/orange/  + labels.csv

Only rows whose image file exists on disk are written to the per-fruit labels.csv
(augmented filenames in the master CSV that were never saved to disk are skipped).
"""
import csv
import shutil
from collections import defaultdict
from pathlib import Path

DATASET_DIR    = Path("dataset")
ALL_IMAGES_DIR = DATASET_DIR / "all_images"
MASTER_CSV     = Path("fruit_labels.csv")


def infer_fruit(filename: str) -> str | None:
    """Map filename to fruit name based on the naming convention in the dataset."""
    f = filename.lower()
    if "apple" in f:
        return "apple"
    if "banana" in f:
        return "banana"
    if "orange" in f:
        return "orange"
    return None


def main() -> None:
    if not ALL_IMAGES_DIR.exists():
        raise SystemExit(f"Source directory not found: {ALL_IMAGES_DIR}")
    if not MASTER_CSV.exists():
        raise SystemExit(f"Master CSV not found: {MASTER_CSV}")

    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    print(f"Master CSV rows  : {len(all_rows)}")

    by_fruit: dict[str, list[dict]] = defaultdict(list)
    skipped_missing = 0
    skipped_unknown = 0

    for row in all_rows:
        fname = row["filename"]
        src   = ALL_IMAGES_DIR / fname
        if not src.exists():
            skipped_missing += 1
            continue
        fruit = infer_fruit(fname)
        if fruit is None:
            skipped_unknown += 1
            print(f"  [WARN] Cannot infer fruit from: {fname}")
            continue
        by_fruit[fruit].append(row)

    found_total = sum(len(v) for v in by_fruit.values())
    print(f"Files found      : {found_total}")
    print(f"Skipped (no file): {skipped_missing}")
    print(f"Skipped (unknown): {skipped_unknown}")
    print()

    for fruit, rows in sorted(by_fruit.items()):
        fruit_dir = DATASET_DIR / fruit
        fruit_dir.mkdir(exist_ok=True)

        # Copy images into the fruit subfolder
        for row in rows:
            src = ALL_IMAGES_DIR / row["filename"]
            dst = fruit_dir / row["filename"]
            if not dst.exists():
                shutil.copy2(src, dst)

        # Write per-fruit labels.csv (filename relative to the fruit folder)
        labels_path = fruit_dir / "labels.csv"
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "ripeness_pct", "days_to_ripe"])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "filename"   : row["filename"],
                    "ripeness_pct": row["ripeness_pct"],
                    "days_to_ripe": row["days_to_ripe"],
                })

        print(f"  dataset/{fruit}/  ->  {len(rows)} images  +  labels.csv written")

    print()
    print("Migration complete.")
    print("Verify the new folders look correct, then you can delete dataset/all_images/")
    print("and archive fruit_labels.csv.")


if __name__ == "__main__":
    main()
