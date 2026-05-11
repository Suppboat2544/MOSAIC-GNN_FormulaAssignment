from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # BACKEND_DIR = platform/backend/  (parents[2] from app/core/config.py)
    BACKEND_DIR: Path = Path(__file__).resolve().parents[2]

    # Project root — kept for backward-compat env overrides; not used for
    # checkpoint / package lookups (those are now inside BACKEND_DIR).
    ROOT_DIR: Path = Path(__file__).resolve().parents[4]

    MODEL_DIR: Path = BACKEND_DIR / "mosaic_gnn"
    OUTPUT_DIR: Path = BACKEND_DIR / "results"
    UPLOAD_DIR: Path = Path("/tmp/mosaic_uploads")
    CONFIG_FILE: Path = BACKEND_DIR / "configs" / "default_config.yaml"

    # Checkpoint paths (relative to BACKEND_DIR)
    CKPT_MODEL9:   str = "checkpoints/Model9/best.pt"
    CKPT_MODEL10:  str = "checkpoints/Model10/best.pt"
    CKPT_MODEL11:  str = "checkpoints/Model11/best.pt"
    CKPT_MODEL11B: str = "checkpoints/Model11b/best.pt"  # PFAS fine-tuned

    # Model
    DEFAULT_MODEL: str = "Model10"
    AVAILABLE_MODELS: list[str] = ["Model9", "Model10", "Model11", "Model11b"]

    # API
    MAX_UPLOAD_SIZE_MB: int = 200

    class Config:
        env_file = ".env"


settings = Settings()
