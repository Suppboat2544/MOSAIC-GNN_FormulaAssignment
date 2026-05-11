"""
Results endpoint — returns per-mass-range and per-element-class formula assignments.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.services.results import load_results, filter_by_mass_range, group_by_element_class

router = APIRouter()


@router.get("/{job_id}")
async def get_results(
    job_id: str,
    mz_min: Optional[float] = Query(None),
    mz_max: Optional[float] = Query(None),
    element_class: Optional[str] = Query(None),
):
    data = load_results(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Results not found")

    if mz_min is not None and mz_max is not None:
        data = filter_by_mass_range(data, mz_min, mz_max)

    if element_class:
        data = [r for r in data if r["element_class"] == element_class]

    return {"job_id": job_id, "count": len(data), "results": data}


@router.get("/{job_id}/element_classes")
async def element_class_summary(job_id: str):
    data = load_results(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Results not found")
    return group_by_element_class(data)


@router.get("/{job_id}/download")
async def download_csv(job_id: str):
    from fastapi.responses import FileResponse
    from app.services.results import export_csv
    path = export_csv(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Results not ready")
    return FileResponse(path, media_type="text/csv", filename=f"mosaic_{job_id}.csv")
