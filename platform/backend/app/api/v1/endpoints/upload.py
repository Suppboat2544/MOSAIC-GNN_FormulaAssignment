"""
Upload endpoint — accepts CSV or mzML files and returns a job_id.
"""
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from app.core.config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx", ".xls",
    ".tsv", ".txt",
    ".mzml", ".mzxml",
}


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    if suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: CSV, XLSX, XLS, TSV, TXT, mzML, mzXML",
        )

    job_id = str(uuid.uuid4())
    dest_dir = settings.UPLOAD_DIR / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"job_id": job_id, "filename": file.filename}


@router.get("/{job_id}/columns")
async def get_columns(job_id: str):
    """Return column names of an uploaded tabular file so the UI can let the
    user choose which column contains the m/z values."""
    upload_dir = settings.UPLOAD_DIR / job_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    files = [f for f in upload_dir.iterdir() if f.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No file uploaded")

    fpath = files[0]
    suffix = fpath.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(fpath, nrows=0)
        elif suffix in (".tsv", ".txt"):
            try:
                df = pd.read_csv(fpath, sep="\t", nrows=0)
            except Exception:
                df = pd.read_csv(fpath, sep=r"\s+", engine="python", nrows=0)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(fpath, nrows=0)
        else:
            # mzML/mzXML — no user-selectable columns
            return {"columns": [], "file_type": "spectrum"}

        return {"columns": df.columns.tolist(), "file_type": "tabular"}

    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read columns: {exc}")
