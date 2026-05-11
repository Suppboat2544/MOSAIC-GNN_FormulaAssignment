"""
Model comparison endpoint — side-by-side metrics for Model9 / Model10 / Model11.
"""
from fastapi import APIRouter, HTTPException
from app.services.compare import load_comparison_metrics

router = APIRouter()


@router.get("/{job_id}")
async def compare_models(job_id: str):
    metrics = load_comparison_metrics(job_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Comparison data not found")
    return metrics
