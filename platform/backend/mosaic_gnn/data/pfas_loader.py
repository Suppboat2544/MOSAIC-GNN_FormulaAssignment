"""
PFAS Dataset Loader — 21T FT-ICR MS AFFF Analysis
===================================================
Loads and converts PFAS data from the Young et al. (2022) ES&T
supplementary Excel file into the MOSAIC-GNN pipeline format.

Source: "PFAS Analysis with Ultrahigh Resolution 21T FT-ICR MS:
        Suspect and Nontargeted Screening with Unrivaled Mass
        Resolving Power and Accuracy"
        DOI: 10.1021/acs.est.1c08143

Sheets used:
  Table S3  — Suspect screening PFAS (163 entries, G/Y/R confidence)
  Table S12 — Nontarget unique formulas (134 entries)
  Table S16 — Nontarget non-unique formulas (567 entries, multiple
              candidates per m/z — perfect for training!)
  Table S5  — Surfactant formulas (48 entries, CHONS only)
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from mosaic_gnn.data.data_pipeline import (
    formula_to_mass, compute_dbe, get_elemental_class, get_dbe_bin,
    parse_formula,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────

PFAS_DATASET_DIR = (
    "dataset/dataset_PFAS Analysis with Ultrahigh Resolution 21T FT-ICR MS"
    ": Suspect and Nontargeted Screening with Unrivaled Mass Resolving Power"
    " and Accuracy"
)
PFAS_XLSX = "es1c08143_si_002.xlsx"


def _find_pfas_xlsx(data_root: str) -> Optional[str]:
    """Locate the PFAS Excel file relative to workspace root."""
    # Try direct path
    direct = os.path.join(data_root, PFAS_DATASET_DIR, PFAS_XLSX)
    if os.path.isfile(direct):
        return direct

    # Try relative to data_root parent
    parent = os.path.dirname(data_root)
    alt = os.path.join(parent, PFAS_DATASET_DIR, PFAS_XLSX)
    if os.path.isfile(alt):
        return alt

    # Globbing fallback
    import glob
    pattern = os.path.join(data_root, '**', 'es1c08143_si_002.xlsx')
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    pattern2 = os.path.join(parent, '**', 'es1c08143_si_002.xlsx')
    matches2 = glob.glob(pattern2, recursive=True)
    if matches2:
        return matches2[0]

    return None


# ─────────────────────────────────────────────────────────────
# Sheet Parsers
# ─────────────────────────────────────────────────────────────

def _build_formula_from_atoms(row: pd.Series, atom_cols: List[str]) -> str:
    """Reconstruct formula string from individual atom count columns."""
    parts = []
    for elem in ['C', 'H', 'N', 'O', 'F', 'Cl', 'S', 'P', 'I']:
        if elem in atom_cols:
            count = int(row.get(elem, 0) or 0)
            if count > 0:
                parts.append(f"{elem}{count}" if count > 1 else elem)
    return ''.join(parts)


def load_pfas_suspect_table(xlsx_path: str) -> pd.DataFrame:
    """
    Load Table S3 — Suspect screening PFAS formulas.

    These have confidence levels (G=green/best, Y=yellow, R=red/least).
    Only includes entries where a single formula is assigned per m/z,
    so every row is a (m/z, formula) pair with high confidence.

    Returns DataFrame in MOSAIC-GNN pipeline format:
        columns: mz_exp, formula, mz_calc, ppm_error, intensity,
                 mass_exact, dbe, elemental_class, dbe_bin,
                 confidence_level, source_sheet
    """
    df = pd.read_excel(xlsx_path, sheet_name='Table S3', header=1)

    # Clean
    df = df.dropna(subset=['Formula', 'Exper. m/z']).copy()

    out = pd.DataFrame({
        'mz_exp': df['Exper. m/z'].astype(float),
        'formula': df['Formula'].astype(str),
        'mz_calc': df['Theor. Mass'].astype(float),
        'ppm_error': df['ppm error'].astype(float),
        'intensity': df['Rel. Abund.'].astype(float),
        'confidence_level': df['Level'].astype(str),
    })

    out['source_sheet'] = 'S3_suspect'
    out = _enrich_derived_columns(out)

    logger.info(f"  Table S3: {len(out)} PFAS entries "
                f"(G={sum(out['confidence_level']=='G')}, "
                f"Y={sum(out['confidence_level']=='Y')}, "
                f"R={sum(out['confidence_level']=='R')})")
    return out


def load_pfas_nontarget_unique(xlsx_path: str) -> pd.DataFrame:
    """
    Load Table S12 — Nontarget PFAS with unique molecular formulas.

    Each m/z has exactly 1 assigned formula + confidence level.
    """
    df = pd.read_excel(xlsx_path, sheet_name='Table S12', header=1)
    df = df.dropna(subset=['Formula', 'Exper. m/z']).copy()

    out = pd.DataFrame({
        'mz_exp': df['Exper. m/z'].astype(float),
        'formula': df['Formula'].astype(str),
        'mz_calc': df['Theor. Mass'].astype(float),
        'ppm_error': df['ppm error'].astype(float),
        'intensity': df['Rel. Abund.'].astype(float),
        'confidence_level': df['Level'].astype(str),
    })

    out['source_sheet'] = 'S12_nontarget_unique'
    out = _enrich_derived_columns(out)

    logger.info(f"  Table S12: {len(out)} nontarget-unique entries "
                f"(G={sum(out['confidence_level']=='G')}, "
                f"Y={sum(out['confidence_level']=='Y')}, "
                f"R={sum(out['confidence_level']=='R')})")
    return out


def load_pfas_nontarget_ambiguous(xlsx_path: str) -> pd.DataFrame:
    """
    Load Table S16 — Nontarget PFAS with non-unique formulas.

    **Key for training**: Multiple formula candidates map to the same m/z.
    This is exactly the problem MOSAIC-GNN solves — disambiguating
    candidate formulas.

    Table S16 has no confidence level column. We group by m/z to identify
    which features have multiple candidates.
    """
    df = pd.read_excel(xlsx_path, sheet_name='Table S16', header=1)
    df = df.dropna(subset=['Formula', 'Exper. m/z']).copy()

    out = pd.DataFrame({
        'mz_exp': df['Exper. m/z'].astype(float),
        'formula': df['Formula'].astype(str),
        'mz_calc': df['Theor. Mass'].astype(float),
        'ppm_error': df['ppm error'].astype(float),
        'intensity': df['Rel. Abund.'].astype(float),
    })

    out['confidence_level'] = 'ambiguous'
    out['source_sheet'] = 'S16_nontarget_ambiguous'
    out = _enrich_derived_columns(out)

    # Stats
    mz_counts = out.groupby('mz_exp').size()
    n_unique_mz = len(mz_counts)
    n_multi = (mz_counts > 1).sum()
    logger.info(f"  Table S16: {len(out)} entries across {n_unique_mz} unique m/z "
                f"({n_multi} m/z with multiple candidates)")
    return out


def load_pfas_surfactants(xlsx_path: str) -> pd.DataFrame:
    """
    Load Table S5 — Surfactant formulas (non-PFAS background).

    These CHONS compounds were co-detected in the AFFF sample.
    Useful as negative examples for PFAS-specific training.
    """
    df = pd.read_excel(xlsx_path, sheet_name='Table S5', header=1)
    df = df.dropna(subset=['Formula', 'Exper. m/z']).copy()

    out = pd.DataFrame({
        'mz_exp': df['Exper. m/z'].astype(float),
        'formula': df['Formula'].astype(str),
        'mz_calc': df['Theor. Mass'].astype(float),
        'ppm_error': df['ppm error'].astype(float),
        'intensity': df['Rel. Abund.'].astype(float),
    })

    out['confidence_level'] = 'surfactant'
    out['source_sheet'] = 'S5_surfactant'
    out = _enrich_derived_columns(out)

    logger.info(f"  Table S5: {len(out)} surfactant entries")
    return out


def _enrich_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns required by the pipeline."""
    df['mass_exact'] = df['formula'].apply(formula_to_mass)
    df['dbe'] = df['formula'].apply(compute_dbe)
    df['elemental_class'] = df['formula'].apply(get_elemental_class)
    df['dbe_bin'] = df['dbe'].apply(get_dbe_bin)
    return df


# ─────────────────────────────────────────────────────────────
# High-Level Loading & Sample Construction
# ─────────────────────────────────────────────────────────────

def load_pfas_data(
    xlsx_path: str,
    min_confidence: str = 'Y',
    include_surfactants: bool = True,
    include_ambiguous: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load all PFAS data and split into logical samples.

    Confidence filter for S3/S12:
        'G' = only green (most confident)
        'Y' = yellow + green
        'R' = all (including red/least confident)

    Returns:
        datasets: Dict[sample_id → DataFrame]
        metadata: DataFrame with sample-level info
    """
    logger.info(f"Loading PFAS dataset from: {xlsx_path}")

    # Load sheets
    s3 = load_pfas_suspect_table(xlsx_path)
    s12 = load_pfas_nontarget_unique(xlsx_path)
    s16 = load_pfas_nontarget_ambiguous(xlsx_path)

    # Confidence filtering
    confidence_order = {'G': 3, 'Y': 2, 'R': 1}
    min_conf_val = confidence_order.get(min_confidence, 1)

    s3_filtered = s3[s3['confidence_level'].map(
        lambda x: confidence_order.get(x, 0) >= min_conf_val)]
    s12_filtered = s12[s12['confidence_level'].map(
        lambda x: confidence_order.get(x, 0) >= min_conf_val)]

    logger.info(f"  After confidence filter (>= {min_confidence}):")
    logger.info(f"    S3:  {len(s3_filtered)}/{len(s3)} entries")
    logger.info(f"    S12: {len(s12_filtered)}/{len(s12)} entries")

    datasets = {}
    metadata_rows = []

    # ── Sample 1: PFAS Suspect (S3) — high confidence, single formula per m/z ──
    # Split G-level into separate sample for training, Y+R for validation
    s3_green = s3_filtered[s3_filtered['confidence_level'] == 'G'].copy()
    s3_rest = s3_filtered[s3_filtered['confidence_level'] != 'G'].copy()

    if len(s3_green) > 0:
        # Deduplicate: keep one formula per m/z (first/highest confidence)
        s3_green_dedup = s3_green.drop_duplicates(subset=['mz_exp'], keep='first')
        s3_green_dedup = s3_green_dedup.reset_index(drop=True)
        datasets['PFAS_AFFF_Suspect_Green'] = s3_green_dedup
        metadata_rows.append({
            'sample_id': 'PFAS_AFFF_Suspect_Green',
            'source': 'PFAS_21T_suspect',
            'matrix_type': 'PFAS_AFFF',
            'instrument': 'FT-ICR_21T',
            'resolution': '21T',
            'polarity': 'negative',
            'num_features': len(s3_green_dedup),
            'has_formula': True,
        })

    if len(s3_rest) > 0:
        s3_rest_dedup = s3_rest.drop_duplicates(subset=['mz_exp'], keep='first')
        s3_rest_dedup = s3_rest_dedup.reset_index(drop=True)
        datasets['PFAS_AFFF_Suspect_YR'] = s3_rest_dedup
        metadata_rows.append({
            'sample_id': 'PFAS_AFFF_Suspect_YR',
            'source': 'PFAS_21T_suspect',
            'matrix_type': 'PFAS_AFFF',
            'instrument': 'FT-ICR_21T',
            'resolution': '21T',
            'polarity': 'negative',
            'num_features': len(s3_rest_dedup),
            'has_formula': True,
        })

    # ── Sample 2: Nontarget unique (S12) ──
    if len(s12_filtered) > 0:
        s12_green = s12_filtered[s12_filtered['confidence_level'] == 'G'].copy()
        s12_rest = s12_filtered[s12_filtered['confidence_level'] != 'G'].copy()

        if len(s12_green) > 0:
            s12_green_dedup = s12_green.drop_duplicates(subset=['mz_exp'], keep='first')
            s12_green_dedup = s12_green_dedup.reset_index(drop=True)
            datasets['PFAS_AFFF_Nontarget_Green'] = s12_green_dedup
            metadata_rows.append({
                'sample_id': 'PFAS_AFFF_Nontarget_Green',
                'source': 'PFAS_21T_nontarget',
                'matrix_type': 'PFAS_AFFF',
                'instrument': 'FT-ICR_21T',
                'resolution': '21T',
                'polarity': 'negative',
                'num_features': len(s12_green_dedup),
                'has_formula': True,
            })

        if len(s12_rest) > 0:
            s12_rest_dedup = s12_rest.drop_duplicates(subset=['mz_exp'], keep='first')
            s12_rest_dedup = s12_rest_dedup.reset_index(drop=True)
            datasets['PFAS_AFFF_Nontarget_YR'] = s12_rest_dedup
            metadata_rows.append({
                'sample_id': 'PFAS_AFFF_Nontarget_YR',
                'source': 'PFAS_21T_nontarget',
                'matrix_type': 'PFAS_AFFF',
                'instrument': 'FT-ICR_21T',
                'resolution': '21T',
                'polarity': 'negative',
                'num_features': len(s12_rest_dedup),
                'has_formula': True,
            })

    # ── Sample 3: S16 ambiguous — multiple formulas per m/z ──
    # For training, we pick the formula with lowest ppm error per m/z
    # as the "ground truth". The rest become natural hard negatives.
    if include_ambiguous and len(s16) > 0:
        # Best formula per m/z (lowest absolute ppm error)
        s16_sorted = s16.copy()
        s16_sorted['abs_ppm'] = s16_sorted['ppm_error'].abs()
        s16_best = s16_sorted.sort_values('abs_ppm').drop_duplicates(
            subset=['mz_exp'], keep='first')
        s16_best = s16_best.drop(columns=['abs_ppm']).reset_index(drop=True)

        datasets['PFAS_AFFF_Nontarget_Ambiguous'] = s16_best
        metadata_rows.append({
            'sample_id': 'PFAS_AFFF_Nontarget_Ambiguous',
            'source': 'PFAS_21T_nontarget_ambiguous',
            'matrix_type': 'PFAS_AFFF',
            'instrument': 'FT-ICR_21T',
            'resolution': '21T',
            'polarity': 'negative',
            'num_features': len(s16_best),
            'has_formula': True,
        })

    # ── Sample 4: Surfactants (non-PFAS background) ──
    if include_surfactants:
        s5 = load_pfas_surfactants(xlsx_path)
        if len(s5) > 0:
            s5_dedup = s5.drop_duplicates(subset=['mz_exp'], keep='first')
            s5_dedup = s5_dedup.reset_index(drop=True)
            datasets['PFAS_AFFF_Surfactants'] = s5_dedup
            metadata_rows.append({
                'sample_id': 'PFAS_AFFF_Surfactants',
                'source': 'PFAS_21T_surfactant',
                'matrix_type': 'PFAS_AFFF',
                'instrument': 'FT-ICR_21T',
                'resolution': '21T',
                'polarity': 'negative',
                'num_features': len(s5_dedup),
                'has_formula': True,
            })

    metadata = pd.DataFrame(metadata_rows)

    logger.info(f"\n  PFAS dataset loaded: {len(datasets)} samples, "
                f"{sum(len(d) for d in datasets.values())} total features")
    for sid, d in datasets.items():
        logger.info(f"    {sid}: {len(d)} features")

    return datasets, metadata


def load_pfas_combined_sample(
    xlsx_path: str,
    min_confidence: str = 'G',
) -> pd.DataFrame:
    """
    Load a single combined DataFrame of all high-confidence PFAS
    assignments (S3 + S12 green-level) — one formula per m/z.

    This is the simplest interface for quick experiments.
    """
    s3 = load_pfas_suspect_table(xlsx_path)
    s12 = load_pfas_nontarget_unique(xlsx_path)

    confidence_order = {'G': 3, 'Y': 2, 'R': 1}
    min_val = confidence_order.get(min_confidence, 3)

    combined = pd.concat([s3, s12], ignore_index=True)
    combined = combined[combined['confidence_level'].map(
        lambda x: confidence_order.get(x, 0) >= min_val)]
    combined = combined.drop_duplicates(subset=['mz_exp'], keep='first')
    combined = combined.sort_values('mz_exp').reset_index(drop=True)

    logger.info(f"  Combined PFAS sample: {len(combined)} features "
                f"(confidence >= {min_confidence})")
    return combined


# ─────────────────────────────────────────────────────────────
# Statistics & Diagnostics
# ─────────────────────────────────────────────────────────────

def pfas_dataset_summary(xlsx_path: str) -> Dict:
    """Print comprehensive summary statistics for the PFAS dataset."""
    s3 = load_pfas_suspect_table(xlsx_path)
    s12 = load_pfas_nontarget_unique(xlsx_path)
    s16 = load_pfas_nontarget_ambiguous(xlsx_path)
    s5 = load_pfas_surfactants(xlsx_path)

    all_pfas = pd.concat([s3, s12, s16], ignore_index=True)

    # Element distribution
    all_formulas = all_pfas['formula'].unique()
    elements_found = set()
    for f in all_formulas:
        parsed = parse_formula(f)
        elements_found.update(parsed.keys())

    # Mass range
    mz_all = all_pfas['mz_exp']

    # F count distribution
    f_counts = []
    for f in all_pfas['formula']:
        parsed = parse_formula(f)
        f_counts.append(parsed.get('F', 0))

    summary = {
        'n_sheets': 4,
        'n_suspect_S3': len(s3),
        'n_nontarget_unique_S12': len(s12),
        'n_nontarget_ambiguous_S16': len(s16),
        'n_surfactant_S5': len(s5),
        'total_pfas_entries': len(all_pfas),
        'unique_formulas': len(all_formulas),
        'unique_mz': all_pfas['mz_exp'].nunique(),
        'mz_range': (float(mz_all.min()), float(mz_all.max())),
        'elements_detected': sorted(elements_found),
        'F_count_range': (min(f_counts), max(f_counts)),
        'F_count_mean': np.mean(f_counts),
        'elemental_classes': all_pfas['elemental_class'].value_counts().to_dict(),
        's3_confidence': s3['confidence_level'].value_counts().to_dict(),
        's12_confidence': s12['confidence_level'].value_counts().to_dict(),
        'mean_abs_ppm': float(all_pfas['ppm_error'].abs().mean()),
        'median_abs_ppm': float(all_pfas['ppm_error'].abs().median()),
    }
    return summary


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import json

    workspace = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    xlsx = _find_pfas_xlsx(workspace)
    if xlsx is None:
        print("ERROR: PFAS Excel file not found!")
        exit(1)

    print(f"Found: {xlsx}\n")

    # Summary
    summary = pfas_dataset_summary(xlsx)
    print("=== PFAS Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Load structured
    print("\n=== Loading structured samples ===")
    datasets, metadata = load_pfas_data(xlsx, min_confidence='Y')
    print("\nMetadata:")
    print(metadata.to_string(index=False))
