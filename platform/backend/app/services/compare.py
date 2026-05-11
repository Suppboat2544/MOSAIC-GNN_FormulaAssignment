"""
Compare service — load Model9/10/11 comparison metrics for a job.
"""
import json
from pathlib import Path
from app.core.config import settings


def load_comparison_metrics(job_id: str) -> dict | None:
    path = settings.OUTPUT_DIR / job_id / "comparison.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)
