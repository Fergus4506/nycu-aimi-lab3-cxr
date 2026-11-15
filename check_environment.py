"""Check preparation only; never download data, create folds, or train a model."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ["torch", "torchvision", "timm", "scikit-learn", "pandas", "opencv-python", "matplotlib"]
COLUMNS = ["new_filename", "normal", "bacteria", "virus", "COVID-19"]


def check_sample(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected sample CSV columns/order")
        rows = list(reader)
    names = [row["new_filename"] for row in rows]
    if len(rows) != 1182 or len(set(names)) != 1182 or not all(names):
        raise ValueError("Expected 1182 unique non-empty sample filenames")
    if any(None in row or any(row[col] != "" for col in COLUMNS[1:]) for row in rows):
        raise ValueError("Sample must contain exactly five fields and empty predictions")
    return {"rows": len(rows), "columns": COLUMNS, "first_five": names[:5],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "base_prefix": sys.base_prefix, "platform": platform.platform(),
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
        "packages": {}, "errors": [],
    }
    for name in PACKAGES:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["sample"] = check_sample(ROOT / "test_data_sample.csv")
    report["spec_sha256"] = hashlib.sha256((ROOT / "spec.pdf").read_bytes()).hexdigest()
    if report["packages"]["torch"]:
        try:
            import torch
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                report["gpu"] = torch.cuda.get_device_name(0)
                report["bf16_supported"] = torch.cuda.is_bf16_supported()
                x = torch.ones((8, 8), device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=report["bf16_supported"]):
                    loss = (x @ x).sum()
                loss.backward()
                torch.cuda.synchronize()
                if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
                    raise RuntimeError("Non-finite CUDA smoke test")
                report["cuda_forward_backward"] = "passed"
        except Exception as exc:
            report["errors"].append(f"CUDA check: {type(exc).__name__}: {exc}")
    report["environment_ready"] = (all(report["packages"].values())
        and report.get("cuda_forward_backward") == "passed" and not report["errors"])
    target = ROOT / "outputs" / "preflight"
    target.mkdir(parents=True, exist_ok=True)
    (target / "environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["environment_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
