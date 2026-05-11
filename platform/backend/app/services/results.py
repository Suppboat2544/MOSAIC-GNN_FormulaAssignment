"""
Results service — load, filter, group, and export formula assignment results.
"""
import json
import csv
import io
from pathlib import Path
from app.core.config import settings


def load_results(job_id: str) -> list[dict] | None:
    path = settings.OUTPUT_DIR / job_id / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def filter_by_mass_range(data: list[dict], mz_min: float, mz_max: float) -> list[dict]:
    return [r for r in data if mz_min <= r["mz"] <= mz_max]


def group_by_element_class(data: list[dict]) -> dict:
    groups: dict[str, int] = {}
    for r in data:
        cls = r.get("element_class", "Unknown")
        groups[cls] = groups.get(cls, 0) + 1
    return {"element_classes": [{"class": k, "count": v} for k, v in sorted(groups.items())]}


def export_csv(job_id: str) -> Path | None:
    data = load_results(job_id)
    if data is None:
        return None
    out_path = settings.OUTPUT_DIR / job_id / "results_export.csv"
    if not out_path.exists() and data:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    return out_path if out_path.exists() else None
