"""Validate the official extracted files without EDA, splitting, or training."""
import csv
import hashlib
import json
from pathlib import Path

from check_environment import COLUMNS, ROOT, check_sample

CLASSES = COLUMNS[1:]
EXPECTED = {"train": 4017, "val": 709, "test": 1182}
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_labels(path, expected_count):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: wrong columns/order")
        rows = list(reader)
    names, counts = set(), dict.fromkeys(CLASSES, 0)
    for row in rows:
        name = row["new_filename"]
        if not name or name in names or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"{path.name}: invalid or duplicate filename {name!r}")
        values = [row[c] for c in CLASSES]
        if None in row or any(v not in {"0", "1"} for v in values) or values.count("1") != 1:
            raise ValueError(f"{path.name}: expected exactly one positive label for {name}")
        names.add(name)
        counts[CLASSES[values.index("1")]] += 1
    if len(rows) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} rows, got {len(rows)}")
    return names, counts


def main():
    raw = ROOT / "data" / "raw"
    report = {"data_root": str(raw), "ready": False, "checks": {}, "errors": [],
              "scope": "File/label inventory only; image decoding and EDA not performed."}
    all_names = {}
    check_sample(ROOT / "test_data_sample.csv")
    for split, expected_count in EXPECTED.items():
        folder = raw / f"{split}_images"
        try:
            if not folder.is_dir():
                raise ValueError(f"Missing directory: data/raw/{split}_images")
            paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
            actual = {p.name for p in paths}
            if len(paths) != len(actual):
                raise ValueError(f"{split}: duplicate image basenames")
            if split == "test":
                with (ROOT / "test_data_sample.csv").open(encoding="utf-8-sig", newline="") as f:
                    expected = {row["new_filename"] for row in csv.DictReader(f)}
                counts = None
            else:
                label_path = raw / f"{split}_data.csv"
                expected, counts = validate_labels(label_path, expected_count)
            if len(paths) != expected_count or actual != expected:
                raise ValueError(f"{split}: images={len(paths)}, expected={expected_count}, "
                                 f"missing={len(expected-actual)}, extra={len(actual-expected)}")
            if any(p.stat().st_size == 0 for p in paths):
                raise ValueError(f"{split}: zero-byte image")
            all_names[split] = actual
            report["checks"][split] = {"images": len(paths), "label_counts": counts}
            if split != "test":
                report["checks"][split]["labels_sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        except (ValueError, OSError, KeyError) as exc:
            report["errors"].append(str(exc))
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = all_names.get(left, set()) & all_names.get(right, set())
        if overlap:
            report["errors"].append(f"{left}/{right}: {len(overlap)} overlapping filenames")
    report["ready"] = not report["errors"]
    output = ROOT / "outputs" / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
