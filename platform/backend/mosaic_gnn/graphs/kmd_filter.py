"""
P1-T2b: KMD Pre-filter — Kendrick Mass Defect Series Binning
=============================================================
Computes Kendrick Mass Defect (KMD) for CH2 and CF2 base units to:
  1. Group features into homologous series.
  2. Assign element-set labels ('PFAS', 'CHO_series', 'matrix_prior').
  3. Reduce formula candidate search space by restricting which
     elements are enumerated per feature.

CH2 KMD groups DOM-type features into methylene series.
CF2 KMD groups PFAS-type features into perfluoroalkyl series.

Output:
  - per-feature KMD values, bin assignments, series group IDs
  - allowed_elements label per feature for G_FCL gating
  - candidate reduction statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# KMD Base Definitions
# ─────────────────────────────────────────────────────────────
KMD_BASES = {
    'CH2': {'nominal': 14, 'exact': 14.01565},
    'CF2': {'nominal': 50, 'exact': 49.99694},
}


def compute_kmd(
    mz_array: np.ndarray,
    base_nominal: int,
    base_exact: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Kendrick Mass and Kendrick Mass Defect for a given base unit.

    KM   = m/z × (nominal / exact)
    KMD  = round(KM) − KM

    Args:
        mz_array:     Array of m/z values.
        base_nominal: Nominal mass of base unit (e.g. 14 for CH2).
        base_exact:   Exact mass of base unit (e.g. 14.01565 for CH2).

    Returns:
        (km, kmd, kmd_bins) — Kendrick mass, KMD, integer bin keys.
    """
    km = mz_array * (base_nominal / base_exact)
    kmd = np.round(km) - km
    # Integer bin key: ±0.001 Da tolerance = ±1 bin unit
    kmd_bins = np.round(kmd * 1000).astype(int)
    return km, kmd, kmd_bins


def assign_homologous_series(
    kmd_bins: np.ndarray,
    min_series_size: int = 2,
    bin_tolerance: int = 1,
) -> np.ndarray:
    """
    Group features into homologous series by KMD bin.

    Features sharing the same KMD bin (±tolerance) and with at least
    min_series_size members are assigned a unique series ID.

    Args:
        kmd_bins:         Integer KMD bin keys per feature.
        min_series_size:  Minimum features to form a series (default: 2).
        bin_tolerance:    Tolerance in bin units (default: 1 = ±0.001 Da).

    Returns:
        series_ids: Array of series IDs per feature (-1 = unassigned).
    """
    N = len(kmd_bins)
    series_ids = np.full(N, -1, dtype=int)

    # Group by bin with tolerance: merge bins within ±tolerance
    sorted_idx = np.argsort(kmd_bins)
    sorted_bins = kmd_bins[sorted_idx]

    current_series = 0
    i = 0
    while i < N:
        # Find contiguous block within tolerance
        j = i + 1
        while j < N and (sorted_bins[j] - sorted_bins[i]) <= 2 * bin_tolerance:
            j += 1

        group_size = j - i
        if group_size >= min_series_size:
            for k in range(i, j):
                series_ids[sorted_idx[k]] = current_series
            current_series += 1

        i = j

    return series_ids


def kmd_prefilter(
    mz_array: np.ndarray,
    default_element_set: str = 'DOM',
) -> Dict[str, np.ndarray]:
    """
    Full KMD pre-filter pipeline.

    For each feature, computes CH2 and CF2 KMD, assigns homologous
    series, and determines the allowed element set for G_FCL enumeration.

    Element set assignment rules:
      - If feature belongs to a CF2 series → 'PFAS' (C,H,O,N,S,F; require F≥2)
      - Elif feature belongs to a CH2 series → 'CHO_series' (C,H,O,N,S only)
      - Else → 'matrix_prior' (use default matrix-specific bounds unchanged)

    Args:
        mz_array:           Array of m/z values (sorted or unsorted).
        default_element_set: Fallback element set for unassigned features.

    Returns:
        Dictionary with:
          - kmd_ch2:          CH2 KMD values [N]
          - kmd_cf2:          CF2 KMD values [N]
          - bin_ch2:          CH2 bin keys [N]
          - bin_cf2:          CF2 bin keys [N]
          - series_id_ch2:    CH2 series group IDs [N] (-1 = unassigned)
          - series_id_cf2:    CF2 series group IDs [N] (-1 = unassigned)
          - allowed_elements: Element set labels [N] ('PFAS'|'CHO_series'|'matrix_prior')
          - km_ch2:           CH2 Kendrick Mass [N] (for plotting)
          - km_cf2:           CF2 Kendrick Mass [N] (for plotting)
    """
    N = len(mz_array)

    # CH2-based KMD
    km_ch2, kmd_ch2, bin_ch2 = compute_kmd(
        mz_array, KMD_BASES['CH2']['nominal'], KMD_BASES['CH2']['exact']
    )
    series_id_ch2 = assign_homologous_series(bin_ch2, min_series_size=2)

    # CF2-based KMD
    km_cf2, kmd_cf2, bin_cf2 = compute_kmd(
        mz_array, KMD_BASES['CF2']['nominal'], KMD_BASES['CF2']['exact']
    )
    series_id_cf2 = assign_homologous_series(bin_cf2, min_series_size=2)

    # Assign element labels
    allowed_elements = np.full(N, 'matrix_prior', dtype=object)

    # CF2 series → PFAS (priority over CH2)
    pfas_mask = series_id_cf2 >= 0
    allowed_elements[pfas_mask] = 'PFAS'

    # CH2 series (not already PFAS) → CHO_series
    cho_mask = (series_id_ch2 >= 0) & (~pfas_mask)
    allowed_elements[cho_mask] = 'CHO_series'

    n_pfas = pfas_mask.sum()
    n_cho = cho_mask.sum()
    n_prior = N - n_pfas - n_cho
    n_ch2_series = len(set(series_id_ch2[series_id_ch2 >= 0]))
    n_cf2_series = len(set(series_id_cf2[series_id_cf2 >= 0]))

    logger.info(
        f"  KMD pre-filter: {N} features → "
        f"PFAS={n_pfas} ({n_cf2_series} CF2 series), "
        f"CHO_series={n_cho} ({n_ch2_series} CH2 series), "
        f"matrix_prior={n_prior}"
    )

    return {
        'kmd_ch2': kmd_ch2,
        'kmd_cf2': kmd_cf2,
        'bin_ch2': bin_ch2,
        'bin_cf2': bin_cf2,
        'series_id_ch2': series_id_ch2,
        'series_id_cf2': series_id_cf2,
        'allowed_elements': allowed_elements,
        'km_ch2': km_ch2,
        'km_cf2': km_cf2,
    }


def get_element_set_for_feature(
    allowed_element: str,
    default_element_set: str = 'DOM',
) -> str:
    """
    Map KMD-assigned element label to FCL element_set key.

    Args:
        allowed_element:    'PFAS', 'CHO_series', or 'matrix_prior'.
        default_element_set: Default element set from sample metadata.

    Returns:
        Element set key for enumerate_candidates().
    """
    if allowed_element == 'PFAS':
        return 'PFAS'       # C,H,O,F (with F≥2 constraint)
    elif allowed_element == 'CHO_series':
        return 'CHONS'      # C,H,O,N,S only (exclude halogens)
    else:
        return default_element_set


def compute_reduction_stats(
    n_before: np.ndarray,
    n_after: np.ndarray,
    allowed_elements: np.ndarray,
) -> pd.DataFrame:
    """
    Compute candidate reduction statistics per feature.

    Args:
        n_before:         Candidate counts before KMD filtering.
        n_after:          Candidate counts after KMD filtering.
        allowed_elements: Element set labels per feature.

    Returns:
        DataFrame with columns:
          feature_idx, n_before, n_after, reduction_pct, element_label
    """
    reduction = np.where(
        n_before > 0,
        100.0 * (1.0 - n_after / n_before),
        0.0,
    )

    df = pd.DataFrame({
        'feature_idx': np.arange(len(n_before)),
        'n_before': n_before.astype(int),
        'n_after': n_after.astype(int),
        'reduction_pct': np.round(reduction, 1),
        'element_label': allowed_elements,
    })

    # Summary statistics
    for label in ['PFAS', 'CHO_series', 'matrix_prior']:
        mask = df['element_label'] == label
        if mask.any():
            mean_red = df.loc[mask, 'reduction_pct'].mean()
            logger.info(f"    {label}: mean reduction = {mean_red:.1f}%  ({mask.sum()} features)")

    return df
