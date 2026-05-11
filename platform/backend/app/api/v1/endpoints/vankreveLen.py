"""
Van Krevelen plot data endpoint — returns H/C vs O/C ratios per formula class.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.services.results import load_results

router = APIRouter()


@router.get("/{job_id}")
async def vankrevelen_data(
    job_id: str,
    element_class: Optional[str] = Query(None),
    mz_min: Optional[float] = Query(None),
    mz_max: Optional[float] = Query(None),
):
    data = load_results(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Results not found")

    if element_class:
        data = [r for r in data if r["element_class"] == element_class]
    if mz_min is not None and mz_max is not None:
        data = [r for r in data if mz_min <= r["mz"] <= mz_max]

    points = [
        {
            "mz": r["mz"],
            "formula": r["formula"],
            "element_class": r["element_class"],
            "H_C": r["H_C"],
            "O_C": r["O_C"],
            "N_C": r["N_C"],
            "S_C": r["S_C"],
            "confidence": r["confidence"],
        }
        for r in data
    ]
    return {"job_id": job_id, "count": len(points), "points": points}
