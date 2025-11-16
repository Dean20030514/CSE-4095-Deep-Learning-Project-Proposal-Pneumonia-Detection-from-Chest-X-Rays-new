from pathlib import Path
from collections import defaultdict
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ALT = ROOT / "data_alt" / "chest_xray"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_DIRS = {"__MACOSX"}
SKIP_FILES_EXACT = {".DS_Store", "Thumbs.db"}
SKIP_PREFIX = ("._",)
REQUIRED_CLASSES = {"NORMAL", "PNEUMONIA"}


def list_files(d: Path):
    return [p for p in d.rglob("*") if p.is_file()]


def check_split(root: Path, split: str):
    split_dir = root / split
    issues = []
    counts = {}
    junk = []
    if not split_dir.exists():
        issues.append(f"[MISSING] {split_dir}")
        return counts, issues, junk
    # class dirs
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    class_names = {d.name for d in class_dirs}
    missing_classes = REQUIRED_CLASSES - class_names
    if missing_classes:
        issues.append(f"[MISSING] {split} missing classes: {sorted(missing_classes)}")
    # count per class and find unwanted files
    for cls in class_dirs:
        files = list_files(cls)
        img_files = [p for p in files if p.suffix.lower() in ALLOWED_EXT]
        unwanted = [p for p in files if (p.suffix.lower() not in ALLOWED_EXT) or (p.name in SKIP_FILES_EXACT) or any(p.name.startswith(pref) for pref in SKIP_PREFIX)]
        counts[f"{split}/{cls.name}"] = len(img_files)
        junk.extend(unwanted)
    return counts, issues, junk


def main():
    if not DATA.exists():
        print(f"[ERROR] Data root not found: {DATA}")
        sys.exit(1)

    total_counts = defaultdict(int)
    all_issues = []
    all_junk = []
    for split in ["train", "val", "test"]:
        counts, issues, junk = check_split(DATA, split)
        for k, v in counts.items():
            total_counts[k] += v
        all_issues.extend(issues)
        all_junk.extend(junk)

    # Report
    print("Data summary (data/):")
    for k in sorted(total_counts.keys()):
        print(f"  {k}: {total_counts[k]}")

    # Issues
    if all_issues:
        print("\nPotential issues:")
        for it in all_issues:
            print("  ", it)
    else:
        print("\nNo structural issues detected (required splits/classes present where expected).")

    # Junk
    macosx_dirs = list(DATA.rglob("__MACOSX"))
    if macosx_dirs:
        print(f"\nFound __MACOSX directories: {len(macosx_dirs)} (should be removed)")
    junk_preview = [p for p in all_junk if (p.name in SKIP_FILES_EXACT) or any(p.name.startswith(pref) for pref in SKIP_PREFIX)]
    if junk_preview:
        print("\nFound unwanted files (preview up to 10):")
        for p in junk_preview[:10]:
            print("  ", p.relative_to(ROOT))

    # Optional: alt dataset summary
    if ALT.exists():
        alt_counts = defaultdict(int)
        for split in ["train", "val", "test"]:
            counts, _, _ = check_split(ALT, split)
            for k, v in counts.items():
                alt_counts[k] += v
        if alt_counts:
            print("\nAlt dataset summary (data_alt/chest_xray/):")
            for k in sorted(alt_counts.keys()):
                print(f"  {k}: {alt_counts[k]}")


if __name__ == "__main__":
    main()
