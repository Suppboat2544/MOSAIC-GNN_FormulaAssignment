from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SampleType(str, Enum):
    dom  = "DOM"
    pfas = "PFAS"


class ModelName(str, Enum):
    model9   = "Model9"
    model10  = "Model10"
    model11  = "Model11"
    model11b = "Model11b"  # PFAS fine-tuned, bypass=True


class PredictRequest(BaseModel):
    job_id: str
    model: ModelName = ModelName.model10
    sample_type: SampleType = SampleType.dom
    mass_range: Optional[tuple[float, float]] = Field(None, description="(min_mz, max_mz)")
    ion_mode: Optional[str] = Field(None, description="positive | negative | both")
    mz_column: Optional[str] = Field(None, description="Column name containing m/z values")
    intensity_column: Optional[str] = Field(None, description="Column name containing intensity values")


class FormulaResult(BaseModel):
    mz: float
    formula: str
    element_class: str
    mass_error_ppm: float
    confidence: float
    H_C: float
    O_C: float
    N_C: float
    S_C: float


class JobStatus(BaseModel):
    job_id: str
    status: str            # queued | running | done | failed
    progress: int = 0      # 0–100
    message: Optional[str] = None
