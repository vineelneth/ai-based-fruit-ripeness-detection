#!/usr/bin/env python
"""
Convert a class-labelled fruit image dataset into per-fruit labels.csv files
compatible with train_model.py.

Supported source structures
---------------------------
Flat  (folder name encodes both fruit and class):
    source/
    ├── freshapples/        <- fruit=apple, class=fresh
    ├── rottenapples/
    └── unripe_banana/

Nested fruit -> class:
    source/
    ├── apple/
    │   ├── fresh/
    │   └── rotten/
    └── banana/
        ├── ripe/
        └── overripe/

Nested class -> fruit:
    source/
    ├── fresh/
    │   ├── apple/
    │   └── banana/
    └── rotten/
        ├── apple/
        └── banana/

Usage examples
--------------
# Auto-detect structure, copy images to dataset/
python remap_dataset.py --source /path/to/downloaded/dataset

# Fruits 360 (all images are "ripe" — no class keyword in folder name)
python remap_dataset.py --source /path/to/Fruits360/Training --default-class ripe

# Preview without copying anything
python remap_dataset.py --source /path/to/dataset --dry-run

# Custom destination folder
python remap_dataset.py --source /path/to/dataset --dest /path/to/dataset/
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Ripeness rules ──────────────────────────────────────────────────────────
# Checked in order (most specific first to avoid "ripe" matching inside "overripe").
# Each entry: (keyword, (ripeness_pct_min, ripeness_pct_max), (days_min, days_max))
# Ripeness values are sampled uniformly from the range for each image to add
# label variance — important when the exact ripeness is unknown.
RIPENESS_RULES = [
    ("overripe",    (90, 100), (0.0, 0.5)),
    ("rotten",      (93, 100), (0.0, 0.0)),
    ("spoiled",     (93, 100), (0.0, 0.0)),
    ("bad",         (90, 100), (0.0, 0.5)),
    ("damaged",     (85, 100), (0.0, 1.0)),
    ("not rip",     (5,  35),  (6.0, 12.0)),  # "not ripe" / "not ripened" — must precede "ripe"
    ("ripening",    (40, 65),  (2.0, 5.0)),
    ("unripe",      (5,  35),  (6.0, 12.0)),
    ("raw",         (5,  35),  (6.0, 12.0)),
    ("immature",    (5,  35),  (6.0, 12.0)),
    ("green",       (10, 40),  (4.0, 10.0)),
    ("fresh",       (60, 90),  (1.0, 3.0)),
    ("ripe",        (70, 90),  (0.0, 2.0)),
    ("mature",      (70, 90),  (0.0, 2.0)),
    ("good",        (60, 85),  (1.0, 3.0)),
]

# ── Fruit name aliases ───────────────────────────────────────────────────────
# Maps any alias (or substring) to a canonical fruit name.
# Add new fruits here before running the script on a new dataset.
FRUIT_ALIASES: dict[str, list[str]] = {
    # ── Core fruits ────────────────────────────────────────────────────────
    "apple":        ["apple", "apples"],
    "apricot":      ["apricot", "apricots"],
    "avocado":      ["avocado", "avocados"],
    "banana":       ["banana", "bananas"],
    "blueberry":    ["blueberry", "blueberries"],
    "cantaloupe":   ["cantaloupe", "cantaloupes"],
    "carambola":    ["carambula", "carambola", "starfruit"],
    "cherry":       ["cherry", "cherries"],
    "clementine":   ["clementine", "clementines"],
    "coconut":      ["cocos", "coconut", "coconuts"],
    "date":         ["dates", "date"],
    "fig":          ["fig", "figs"],
    "grape":        ["grape", "grapes"],
    "grapefruit":   ["grapefruit", "grapefruits"],
    "guava":        ["guava", "guavas"],
    "huckleberry":  ["huckleberry", "huckleberries"],
    "kaki":         ["kaki", "persimmon", "persimmons"],
    "kiwi":         ["kiwi", "kiwis", "kiwifruit"],
    "kumquat":      ["kumquats", "kumquat"],
    "lemon":        ["lemon", "lemons"],
    "lime":         ["lime", "limes"],
    "lychee":       ["lychee", "lychees"],
    "mandarin":     ["mandarine", "mandarin", "mandarins", "tangelo", "tangelos"],
    "mango":        ["mango", "mangoes", "mangos"],
    "mangosteen":   ["mangostan", "mangosteen", "mangosteens"],
    "melon":        ["melon", "melons"],
    "mulberry":     ["mulberry", "mulberries"],
    "nectarine":    ["nectarine", "nectarines"],
    "orange":       ["orange", "oranges"],
    "papaya":       ["papaya", "papayas", "pawpaw"],
    "passion_fruit":["maracuja", "passion fruit", "granadilla", "passionfruit"],
    "peach":        ["peach", "peaches"],
    "pear":         ["pear", "pears"],
    "physalis":     ["physalis"],
    "pineapple":    ["pineapple", "pineapples"],
    "pitahaya":     ["pitahaya", "dragonfruit", "dragon fruit"],
    "plum":         ["plum", "plums"],
    "pomegranate":  ["pomegranate", "pomegranates"],
    "pomelo":       ["pomelo", "pomelos", "sweetie"],
    "quince":       ["quince", "quinces"],
    "rambutan":     ["rambutan", "rambutans"],
    "raspberry":    ["raspberry", "raspberries"],
    "redcurrant":   ["redcurrant", "redcurrants"],
    "salak":        ["salak"],
    "strawberry":   ["strawberry", "strawberries"],
    "tamarillo":    ["tamarillo", "tamarillos"],
    "tomato":       ["tomato", "tomatoes"],
    "watermelon":   ["watermelon", "watermelons"],
    # ── Fruit-vegetables (have meaningful ripeness stages) ─────────────────
    "cucumber":     ["cucumber", "cucumbers"],
    "pepper":       ["pepper", "peppers"],
    "zucchini":     ["zucchini", "zucchinis", "courgette"],
    "eggplant":     ["eggplant", "eggplants", "aubergine"],
    "pepino":       ["pepino"],
    # ── Nuts (skip by omission — no ripeness concept) ──────────────────────
    # Hazelnut, Walnut, Chestnut, Nut Pecan, Nut Forest → not listed → SKIP
    # ── Root vegetables / staples (skip) ───────────────────────────────────
    # Beetroot, Cabbage, Carrot, Cauliflower, Corn, Ginger Root,
    # Kohlrabi, Onion, Potato → not listed → SKIP
}

# Build reverse lookup: alias → canonical name
_ALIAS_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in FRUIT_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_LOOKUP[_alias.lower()] = _canonical

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_fruit(name: str) -> str | None:
    """Return the canonical fruit name for a folder name, or None if unknown."""
    name_lower = name.lower().strip()

    # 1. Exact match
    if name_lower in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[name_lower]

    # 2. First-word match — handles "Watermelon 1" (not "melon"),
    #    "Tomato Cherry Red 1" (not "cherry"), "Mangostan 1" (not "mango")
    first_word = name_lower.split()[0] if name_lower else ""
    if first_word in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[first_word]

    # 3. Longest-alias substring match — handles "freshapples", "rottenbanana"
    #    Longest-first prevents "melon" stealing from "watermelon" in flat names
    for alias in sorted(_ALIAS_LOOKUP.keys(), key=len, reverse=True):
        if alias in name_lower:
            return _ALIAS_LOOKUP[alias]

    return None


def classify_name(name: str) -> tuple[str | None, tuple | None]:
    """
    Match a folder name against RIPENESS_RULES.
    Returns (keyword, ((rip_min, rip_max), (days_min, days_max))) or (None, None).
    """
    name_lower = name.lower()
    for keyword, rip_range, days_range in RIPENESS_RULES:
        if keyword in name_lower:
            return keyword, (rip_range, days_range)
    return None, None


def sample_label(rip_range: tuple, days_range: tuple, rng: random.Random) -> tuple[float, float]:
    return (
        round(rng.uniform(*rip_range), 1),
        round(rng.uniform(*days_range), 1),
    )


def list_images(folder: Path) -> list[Path]:
    return [f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]


def detect_structure(source: Path) -> str:
    """
    Auto-detect whether the source dataset is flat, nested-fruit-class, or nested-class-fruit.
    """
    top_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not top_dirs:
        return "flat"

    # If top-level dirs contain further subdirectories, it is nested
    sample = top_dirs[:5]
    any_nested = any(any(c.is_dir() for c in d.iterdir()) for d in sample)
    if not any_nested:
        return "flat"

    # Determine orientation by counting fruit vs class keyword hits at top level
    fruit_hits = sum(1 for d in sample if normalize_fruit(d.name) is not None)
    class_hits = sum(1 for d in sample if classify_name(d.name)[0] is not None)

    return "nested-fruit-class" if fruit_hits >= class_hits else "nested-class-fruit"


# ── Collection ───────────────────────────────────────────────────────────────

def collect_flat(source: Path, default_ranges, rng) -> dict[str, list]:
    result = defaultdict(list)
    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        fruit = normalize_fruit(folder.name)
        _, ranges = classify_name(folder.name)
        if ranges is None:
            ranges = default_ranges
        if fruit is None or ranges is None:
            print(f"  [SKIP] {folder.name}/  -- cannot determine fruit or class")
            continue
        images = list_images(folder)
        for img in images:
            result[fruit].append((img, *sample_label(*ranges, rng)))
        print(f"  [MAP]  {folder.name}/  ->  fruit={fruit}  ({len(images)} images)")
    return result


def collect_nested_fruit_class(source: Path, default_ranges, rng) -> dict[str, list]:
    result = defaultdict(list)
    for fruit_dir in sorted(source.iterdir()):
        if not fruit_dir.is_dir():
            continue
        fruit = normalize_fruit(fruit_dir.name)
        if fruit is None:
            print(f"  [SKIP] {fruit_dir.name}/  -- unrecognized fruit name")
            continue
        class_dirs = [d for d in fruit_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            # Leaf folder with no class subdirs — use default class
            _, ranges = None, default_ranges
            if ranges is None:
                print(f"  [SKIP] {fruit_dir.name}/  -- no class subfolders and no --default-class")
                continue
            images = list_images(fruit_dir)
            for img in images:
                result[fruit].append((img, *sample_label(*ranges, rng)))
            print(f"  [MAP]  {fruit_dir.name}/  ->  fruit={fruit}  ({len(images)} images)")
        else:
            for class_dir in sorted(class_dirs):
                _, ranges = classify_name(class_dir.name)
                if ranges is None:
                    ranges = default_ranges
                if ranges is None:
                    print(f"  [SKIP] {fruit_dir.name}/{class_dir.name}/  -- unrecognized class")
                    continue
                images = list_images(class_dir)
                for img in images:
                    result[fruit].append((img, *sample_label(*ranges, rng)))
                print(f"  [MAP]  {fruit_dir.name}/{class_dir.name}/  ->  fruit={fruit}  ({len(images)} images)")
    return result


def collect_nested_class_fruit(source: Path, default_ranges, rng) -> dict[str, list]:
    result = defaultdict(list)
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        _, ranges = classify_name(class_dir.name)
        if ranges is None:
            ranges = default_ranges
        if ranges is None:
            print(f"  [SKIP] {class_dir.name}/  -- unrecognized class and no --default-class")
            continue
        for fruit_dir in sorted(class_dir.iterdir()):
            if not fruit_dir.is_dir():
                continue
            fruit = normalize_fruit(fruit_dir.name)
            if fruit is None:
                print(f"  [SKIP] {class_dir.name}/{fruit_dir.name}/  -- unrecognized fruit")
                continue
            images = list_images(fruit_dir)
            for img in images:
                result[fruit].append((img, *sample_label(*ranges, rng)))
            print(f"  [MAP]  {class_dir.name}/{fruit_dir.name}/  ->  fruit={fruit}  ({len(images)} images)")
    return result


# ── Writing ──────────────────────────────────────────────────────────────────

def write_fruit(fruit: str, entries: list, dest: Path, dry_run: bool) -> None:
    fruit_dir = dest / fruit
    labels_path = fruit_dir / "labels.csv"

    # Load existing labels if the fruit folder already has data
    existing: list[dict] = []
    existing_filenames: set[str] = set()
    if labels_path.exists():
        with open(labels_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
            existing_filenames = {r["filename"] for r in existing}

    new_rows = []
    for src_path, ripeness, days in entries:
        dest_filename = src_path.name
        # Avoid duplicate filenames: suffix with _1, _2, ... as needed
        stem, suffix = src_path.stem, src_path.suffix
        counter = 1
        while dest_filename in existing_filenames or dest_filename in {r["filename"] for r in new_rows}:
            dest_filename = f"{stem}_{counter}{suffix}"
            counter += 1

        dst = fruit_dir / dest_filename
        if not dry_run:
            fruit_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)

        new_rows.append({"filename": dest_filename, "ripeness_pct": ripeness, "days_to_ripe": days})

    if not dry_run:
        all_rows = existing + new_rows
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "ripeness_pct", "days_to_ripe"])
            writer.writeheader()
            writer.writerows(all_rows)

    action = "Would write" if dry_run else "Wrote"
    prior = f" (+{len(existing)} existing)" if existing else ""
    print(f"  {action} dataset/{fruit}/labels.csv  —  {len(new_rows)} new rows{prior}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap a class-labelled fruit dataset to per-fruit labels.csv format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", required=True,
                        help="Root folder of the downloaded dataset")
    parser.add_argument("--dest", default="dataset",
                        help="Destination root (default: dataset/)")
    parser.add_argument("--structure", choices=["auto", "flat", "nested-fruit-class", "nested-class-fruit"],
                        default="auto",
                        help="Dataset folder structure (default: auto-detect)")
    parser.add_argument("--default-class", default=None,
                        help="Fallback class when folder name has no class keyword "
                             "(e.g. 'ripe' for Fruits 360 where all images are ripe)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for label sampling (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without copying any files")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest   = Path(args.dest).resolve()

    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    rng = random.Random(args.seed)

    # Resolve default class ranges
    default_ranges = None
    if args.default_class:
        _, default_ranges = classify_name(args.default_class)
        if default_ranges is None:
            raise SystemExit(
                f"Unrecognized --default-class '{args.default_class}'. "
                f"Valid keywords: {[r[0] for r in RIPENESS_RULES]}"
            )

    structure = args.structure
    if structure == "auto":
        structure = detect_structure(source)
    print(f"\nSource    : {source}")
    print(f"Dest      : {dest}")
    print(f"Structure : {structure}")
    if args.default_class:
        print(f"Default   : {args.default_class}  ->  {default_ranges}")
    if args.dry_run:
        print("Mode      : DRY RUN (no files will be copied)")
    print()

    collectors = {
        "flat":                collect_flat,
        "nested-fruit-class":  collect_nested_fruit_class,
        "nested-class-fruit":  collect_nested_class_fruit,
    }
    by_fruit = collectors[structure](source, default_ranges, rng)

    if not by_fruit:
        raise SystemExit("\nNo images collected. Check --source and --structure.")

    print()
    for fruit in sorted(by_fruit):
        write_fruit(fruit, by_fruit[fruit], dest, dry_run=args.dry_run)

    total = sum(len(v) for v in by_fruit.values())
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done — {total} images across {len(by_fruit)} fruit(s).")
    if args.dry_run:
        print("Remove --dry-run to apply.")


if __name__ == "__main__":
    main()
