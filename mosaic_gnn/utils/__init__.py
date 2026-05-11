"""
MOSAIC-GNN utility functions.
Reproducibility, configuration, and common helpers.
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Device-agnostic: GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'configs', 'default_config.yaml'
        )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def ppm_to_da(mz: float, ppm: float) -> float:
    """Convert ppm mass error to absolute Dalton error."""
    return mz * ppm / 1e6


def da_to_ppm(mz: float, da: float) -> float:
    """Convert absolute Dalton error to ppm."""
    return da / mz * 1e6


def compute_mass_tolerance(mz_i: float, mz_j: float, ppm_error: float = 3.0) -> float:
    """
    Compute combined mass tolerance for a pair of features.
    σ_ij = sqrt(σ_i² + σ_j²), where σ = m/z × ppm / 1e6.
    Returns 3σ_ij (3-sigma tolerance).
    """
    sigma_i = mz_i * ppm_error / 1e6
    sigma_j = mz_j * ppm_error / 1e6
    return 3 * np.sqrt(sigma_i**2 + sigma_j**2)
