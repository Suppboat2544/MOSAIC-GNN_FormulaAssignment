"""
Unified PFAS Dataset Loader
============================
Combines ALL available PFAS datasets for training:

1. **21T FT-ICR MS** (Young et al., 2022):
   - Tables S3, S5, S12, S16 from es1c08143_si_002.xlsx
   - 6 samples (~456 features total), real instrument data

2. **EPA MasterList** (MSV000085244 / FluoroMatch Core-Match):
   - 2,586 known PFAS compounds with [M-H]- m/z
   - Filtered to elements {C,H,O,N,S,P,F,Cl,I}
   - Split into synthetic samples of ~200 features each

3. **Precursor Library NEG** (MSV000087033 / FluoroMatch):
   - 2,589 PFAS precursor ions (largely overlaps EPA MasterList)
   - Used for deduplication / augmentation

Returns data in the same format as :func:`load_pfas_data`:
    ``(Dict[sample_id → DataFrame], DataFrame)``
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
from mosaic_gnn.data.pfas_loader import (
    load_pfas_data, _find_pfas_xlsx,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Allowed elements for PFAS_AFFF pipeline
# ─────────────────────────────────────────────────────────────
PFAS_ALLOWED_ELEMENTS = frozenset(['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'I'])

# ─────────────────────────────────────────────────────────────
# Dataset directory names
# ─────────────────────────────────────────────────────────────
MSV85244_DIR = 'dataset_MSV000085244_PFAS_MSW_Leachate_Foam'
MSV87033_DIR = 'dataset_MSV000087033_FluoroMatch_Modular_Flow'


def _find_dataset_root(workspace_root: str) -> str:
    """Return path to the dataset/ directory."""
    candidates = [
        os.path.join(workspace_root, 'dataset'),
        os.path.join(os.path.dirname(workspace_root), 'dataset'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(workspace_root, 'dataset')


def _formula_has_only_allowed_elements(formula: str) -> bool:
    """Check if a formula string contains only PFAS-allowed elements."""
    try:
        parsed = parse_formula(formula)
        return set(parsed.keys()).issubset(PFAS_ALLOWED_ELEMENTS)
    except Exception:
        return False


def _enrich_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: mass_exact, dbe, elemental_class, dbe_bin."""
    df = df.copy()
    df['mass_exact'] = df['formula'].apply(formula_to_mass)
    df['dbe'] = df['formula'].apply(compute_dbe)
    df['elemental_class'] = df['formula'].apply(get_elemental_class)
    df['dbe_bin'] = df['dbe'].apply(get_dbe_bin)
    return df


# ─────────────────────────────────────────────────────────────
# EPA MasterList Loader
# ─────────────────────────────────────────────────────────────

def load_epa_masterlist(csv_path: str) -> pd.DataFrame:
    """
    Load EPA PFAS MasterList CSV (Kaufmann Plots edition).

    Columns: row.ID, INCHIKEY, Name, SMILES, Formula, Mass, [M-H]-, ...

    Returns DataFrame with pipeline-standard columns:
        mz_exp, formula, mz_calc, ppm_error, intensity, ...
    """
    logger.info(f"  Loading EPA MasterList: {csv_path}")
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Clean: require Formula and [M-H]- columns
    df = df.dropna(subset=['Formula', '[M-H]-']).copy()
    df = df[df['[M-H]-'].apply(lambda x: isinstance(x, (int, float)) or str(x).replace('.', '').replace('-', '').isdigit())]
    df['[M-H]-'] = df['[M-H]-'].astype(float)
    df['Mass'] = df['Mass'].astype(float)

    # Filter to allowed elements only
    mask = df['Formula'].apply(_formula_has_only_allowed_elements)
    n_before = len(df)
    df = df[mask].copy()
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info(f"    Filtered {n_filtered}/{n_before} entries with "
                    f"disallowed elements (Br, Sn, Si, Se, Hg, B, Ag, etc.)")

    # Build pipeline DataFrame
    out = pd.DataFrame({
        'mz_exp': df['[M-H]-'].values,
        'formula': df['Formula'].astype(str).values,
        'mz_calc': df['[M-H]-'].values,  # calculated, not measured → ppm_error = 0
        'ppm_error': 0.0,
        'intensity': 1.0,  # no measured intensity for library entries
        'confidence_level': 'library',
        'source_sheet': 'EPA_MasterList',
    })

    # Store SMILES for potential future use
    if 'SMILES' in df.columns:
        out['smiles'] = df['SMILES'].values

    # Deduplicate by formula (keep first)
    out = out.drop_duplicates(subset=['formula'], keep='first')
    out = out.reset_index(drop=True)

    out = _enrich_derived_columns(out)

    logger.info(f"    EPA MasterList: {len(out)} unique PFAS formulas "
                f"(m/z range {out['mz_exp'].min():.1f}–{out['mz_exp'].max():.1f})")
    return out


# ─────────────────────────────────────────────────────────────
# Precursor Library NEG Loader
# ─────────────────────────────────────────────────────────────

def load_precursor_library_neg(csv_path: str) -> pd.DataFrame:
    """
    Load FluoroMatch Precursor Library (negative mode).

    Format: "DTXSID;Name;CAS;InChIKey;Formula;SMILES;Adduct",m/z_value
    No header row.
    """
    logger.info(f"  Loading Precursor Library NEG: {csv_path}")

    rows = []
    with open(csv_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on last comma to get m/z
            # Format: "metadata_string",mz_value
            try:
                # Find the last comma that separates metadata from m/z
                last_comma = line.rfind(',')
                if last_comma < 0:
                    continue
                mz_str = line[last_comma + 1:].strip()
                meta_str = line[:last_comma].strip().strip('"')

                mz = float(mz_str)
                parts = meta_str.split(';')
                if len(parts) >= 6:
                    formula = parts[4].strip()
                    smiles = parts[5].strip()
                    name = parts[1].strip()
                    adduct = parts[6].strip() if len(parts) > 6 else '[M-H]-'
                    rows.append({
                        'formula': formula,
                        'mz': mz,
                        'smiles': smiles,
                        'name': name,
                        'adduct': adduct,
                    })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)

    # Only keep [M-H]- adduct entries
    df = df[df['adduct'].str.contains(r'\[M-H\]-', regex=True, na=False)].copy()

    # Filter to allowed elements
    mask = df['formula'].apply(_formula_has_only_allowed_elements)
    n_before = len(df)
    df = df[mask].copy()
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info(f"    Filtered {n_filtered}/{n_before} entries with disallowed elements")

    out = pd.DataFrame({
        'mz_exp': df['mz'].values,
        'formula': df['formula'].astype(str).values,
        'mz_calc': df['mz'].values,
        'ppm_error': 0.0,
        'intensity': 1.0,
        'confidence_level': 'library',
        'source_sheet': 'Precursor_Library_NEG',
    })

    if 'smiles' in df.columns:
        out['smiles'] = df['smiles'].values

    out = out.drop_duplicates(subset=['formula'], keep='first')
    out = out.reset_index(drop=True)

    out = _enrich_derived_columns(out)

    logger.info(f"    Precursor Library: {len(out)} unique PFAS formulas")
    return out


# ─────────────────────────────────────────────────────────────
# Sample Splitting Utility
# ─────────────────────────────────────────────────────────────

def _split_into_samples(
    df: pd.DataFrame,
    prefix: str,
    target_size: int = 200,
    source_name: str = 'EPA_Library',
    shuffle_seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    """
    Split a large DataFrame into multiple manageable samples.

    Sorts by m/z first, then splits into chunks of `target_size`.
    This keeps each sample at a consistent mass range, which is
    more realistic for real-world mass spec runs.
    """
    df_sorted = df.sort_values('mz_exp').reset_index(drop=True)

    n = len(df_sorted)
    n_samples = max(1, (n + target_size - 1) // target_size)

    datasets = {}
    metadata_rows = []

    for i in range(n_samples):
        start = i * target_size
        end = min((i + 1) * target_size, n)
        chunk = df_sorted.iloc[start:end].copy().reset_index(drop=True)

        sample_id = f'{prefix}_batch{i + 1}'
        datasets[sample_id] = chunk
        metadata_rows.append({
            'sample_id': sample_id,
            'source': source_name,
            'matrix_type': 'PFAS_AFFF',
            'instrument': 'library',
            'resolution': 'calculated',
            'polarity': 'negative',
            'num_features': len(chunk),
            'has_formula': True,
        })

    return datasets, metadata_rows


# ─────────────────────────────────────────────────────────────
# Unified Loader
# ─────────────────────────────────────────────────────────────

def load_all_pfas_data(
    workspace_root: str,
    min_confidence: str = 'Y',
    include_surfactants: bool = True,
    include_ambiguous: bool = True,
    include_epa_library: bool = True,
    epa_sample_size: int = 200,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load ALL available PFAS datasets from the workspace.

    Combines:
    1. 21T FT-ICR MS data (6 samples, ~456 features)
    2. EPA MasterList library (split into ~200-feature samples)
    3. Precursor Library NEG (for dedup/augmentation with EPA)

    Parameters
    ----------
    workspace_root : str
        Path to the workspace root directory.
    min_confidence : str
        Minimum confidence level for 21T data ('G', 'Y', or 'R').
    include_surfactants : bool
        Include CHONS surfactant data from Table S5.
    include_ambiguous : bool
        Include ambiguous Table S16 data.
    include_epa_library : bool
        Include EPA MasterList library data.
    epa_sample_size : int
        Target number of features per EPA library sample.

    Returns
    -------
    datasets : Dict[str, pd.DataFrame]
        Sample ID → DataFrame with columns:
        mz_exp, formula, mz_calc, ppm_error, intensity,
        confidence_level, source_sheet, mass_exact, dbe,
        elemental_class, dbe_bin
    metadata : pd.DataFrame
        Sample-level metadata.
    """
    all_datasets = {}
    all_metadata = []

    # ── Part 1: Load 21T FT-ICR MS data ──
    xlsx_path = _find_pfas_xlsx(workspace_root)
    if xlsx_path is not None and os.path.isfile(xlsx_path):
        logger.info("=" * 60)
        logger.info("Part 1: Loading 21T FT-ICR MS data")
        logger.info("=" * 60)
        fticr_datasets, fticr_meta = load_pfas_data(
            xlsx_path,
            min_confidence=min_confidence,
            include_surfactants=include_surfactants,
            include_ambiguous=include_ambiguous,
        )
        all_datasets.update(fticr_datasets)
        all_metadata.extend(fticr_meta.to_dict('records'))
        logger.info(f"  → {len(fticr_datasets)} samples from 21T FT-ICR MS")
    else:
        logger.warning("  21T FT-ICR PFAS Excel not found — skipping")

    # ── Part 2: Load EPA MasterList ──
    if include_epa_library:
        dataset_root = _find_dataset_root(workspace_root)

        # Collect all unique (formula, mz) from EPA + Precursor libraries
        epa_dfs = []

        # Try MSV000085244 directory
        epa_85244 = os.path.join(dataset_root, MSV85244_DIR,
                                  'EPA_MasterList_Kaufmann_Plots.csv')
        if os.path.isfile(epa_85244):
            logger.info("\n" + "=" * 60)
            logger.info("Part 2a: Loading EPA MasterList (MSV000085244)")
            logger.info("=" * 60)
            epa_dfs.append(load_epa_masterlist(epa_85244))

        # Try Precursor Library from MSV000087033
        prec_87033 = os.path.join(dataset_root, MSV87033_DIR,
                                   'Precursor_Library_NEG.csv')
        if os.path.isfile(prec_87033):
            logger.info("\n" + "=" * 60)
            logger.info("Part 2b: Loading Precursor Library NEG (MSV000087033)")
            logger.info("=" * 60)
            epa_dfs.append(load_precursor_library_neg(prec_87033))

        # Also check MSV000085244's copy of Precursor Library
        prec_85244 = os.path.join(dataset_root, MSV85244_DIR,
                                   'Precursor_Library_NEG.csv')
        if os.path.isfile(prec_85244) and prec_85244 != prec_87033:
            logger.info("\n" + "=" * 60)
            logger.info("Part 2c: Loading Precursor Library NEG (MSV000085244)")
            logger.info("=" * 60)
            epa_dfs.append(load_precursor_library_neg(prec_85244))

        if epa_dfs:
            # Merge and deduplicate across all library sources
            combined = pd.concat(epa_dfs, ignore_index=True)

            # Drop columns that may not be present in all sources
            cols_to_keep = ['mz_exp', 'formula', 'mz_calc', 'ppm_error',
                           'intensity', 'confidence_level', 'source_sheet',
                           'mass_exact', 'dbe', 'elemental_class', 'dbe_bin']
            combined = combined[[c for c in cols_to_keep if c in combined.columns]]

            # Deduplicate by formula (prefer EPA over Precursor)
            combined = combined.drop_duplicates(subset=['formula'], keep='first')
            combined = combined.sort_values('mz_exp').reset_index(drop=True)

            # Remove formulas already present in 21T data
            existing_formulas = set()
            for ds in all_datasets.values():
                existing_formulas.update(ds['formula'].unique())
            mask_new = ~combined['formula'].isin(existing_formulas)
            n_overlap = (~mask_new).sum()
            combined = combined[mask_new].reset_index(drop=True)

            logger.info(f"\n  Merged library: {len(combined)} new unique formulas "
                        f"({n_overlap} already in 21T data)")

            if len(combined) > 0:
                # Split into samples
                lib_datasets, lib_metadata = _split_into_samples(
                    combined,
                    prefix='EPA_Library',
                    target_size=epa_sample_size,
                    source_name='EPA_PFAS_Library',
                )
                all_datasets.update(lib_datasets)
                all_metadata.extend(lib_metadata)

                logger.info(f"  → {len(lib_datasets)} library samples "
                            f"({sum(len(d) for d in lib_datasets.values())} total features)")

    # ── Summary ──
    metadata = pd.DataFrame(all_metadata)
    total_features = sum(len(d) for d in all_datasets.values())

    logger.info("\n" + "=" * 60)
    logger.info(f"UNIFIED PFAS DATASET: {len(all_datasets)} samples, "
                f"{total_features} total features")
    logger.info("=" * 60)
    for sid, d in sorted(all_datasets.items()):
        logger.info(f"  {sid:40s} {len(d):5d} features")

    return all_datasets, metadata


# ─────────────────────────────────────────────────────────────
# Convenience: Quick loading for scripts
# ─────────────────────────────────────────────────────────────

def load_unified_pfas(
    workspace_root: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Convenience wrapper: auto-detect workspace and load all PFAS data.
    """
    if workspace_root is None:
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

    return load_all_pfas_data(
        workspace_root=workspace_root,
        min_confidence='Y',
        include_epa_library=True,
        epa_sample_size=200,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    workspace = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(f"Workspace: {workspace}\n")

    datasets, metadata = load_unified_pfas(workspace)

    print(f"\n{'='*60}")
    print("METADATA:")
    print(metadata.to_string(index=False))

    # Verify all DataFrames have required columns
    required_cols = ['mz_exp', 'formula', 'mz_calc', 'ppm_error',
                     'intensity', 'mass_exact', 'dbe', 'elemental_class',
                     'dbe_bin']
    for sid, df in datasets.items():
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {sid} missing columns: {missing}")
        else:
            print(f"  ✓ {sid}: {len(df)} features, all columns present")
