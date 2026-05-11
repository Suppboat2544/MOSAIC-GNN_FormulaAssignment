"""
Predict endpoint — launches MOSAIC-GNN inference for a given job_id.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas.predict import PredictRequest, JobStatus
from app.services.inference import run_inference, get_job_status

router = APIRouter()


@router.post("/", response_model=JobStatus)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_inference, req)
    return JobStatus(job_id=req.job_id, status="queued", progress=0)


@router.get("/{job_id}/status", response_model=JobStatus)
async def status(job_id: str):
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
