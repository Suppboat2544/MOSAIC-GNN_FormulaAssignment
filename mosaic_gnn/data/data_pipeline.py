"""
P1-T1: Raw Data Acquisition & Curation
=======================================
Loads, cleans, and standardizes raw HRMS peak list data from multiple sources:
  - DOM training/testing CSV files (FT-ICR MS)
  - KNN-format tab-separated files (with ion mobility)
  - Raw peak lists (unassigned m/z + intensity)

Outputs:
  - Cleaned DataFrames with standardized columns
  - metadata.csv with sample-level information
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Formula parsing utilities
# ─────────────────────────────────────────────────────────────

# Regex to parse molecular formulas like C19H18O6, C16H13N5O3S
_FORMULA_REGEX = re.compile(r'([A-Z][a-z]?)(\d*)')

# Standard element order
ELEMENTS = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si']

# Exact monoisotopic atomic masses (IUPAC 2020)
ATOMIC_MASSES: Dict[str, float] = {
    'C':  12.000000,
    'H':  1.00782503207,
    'O':  15.99491461956,
    'N':  14.0030740048,
    'S':  31.97207100,
    'P':  30.97376163,
    'F':  18.99840322,
    'Cl': 34.96885268,
    'Br': 78.9183371,
    'I':  126.904473,
    'Si': 27.9769265325,
}

# Electron mass (for [M-H]- calculations)
ELECTRON_MASS = 0.00054858


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula string into element counts.
    
    Args:
        formula: e.g. 'C19H18O6', 'C16H13N5O3'
    
    Returns:
        Dictionary mapping element symbols to counts.
        
    Example:
        >>> parse_formula('C19H18O6')
        {'C': 19, 'H': 18, 'O': 6}
    """
    counts: Dict[str, int] = {}
    for match in _FORMULA_REGEX.finditer(formula):
        elem = match.group(1)
        num = match.group(2)
        if elem:  # Skip empty matches
            counts[elem] = int(num) if num else 1
    return counts


def formula_to_vector(formula: str) -> np.ndarray:
    """
    Convert a formula string to a fixed-length vector [C, H, O, N, S, P, F, Cl, Br].
    
    Args:
        formula: Molecular formula string.
        
    Returns:
        numpy array of shape (9,) with element counts.
    """
    counts = parse_formula(formula)
    return np.array([counts.get(e, 0) for e in ELEMENTS], dtype=np.float32)


def formula_to_mass(formula: str) -> float:
    """
    Compute exact monoisotopic mass from molecular formula.
    
    Args:
        formula: Molecular formula string.
        
    Returns:
        Exact monoisotopic mass in Daltons.
    """
    counts = parse_formula(formula)
    mass = 0.0
    for elem, count in counts.items():
        if elem in ATOMIC_MASSES:
            mass += ATOMIC_MASSES[elem] * count
        else:
            logger.warning(f"Unknown element '{elem}' in formula '{formula}'")
    return mass


def compute_dbe(formula: str) -> float:
    """
    Compute Double Bond Equivalents (DBE / RDBE).
    
    DBE = C - H/2 + N/2 + 1  (for CHONS formulas)
    General: DBE = 1 + Σ(n_i * (v_i - 2)) / 2
    
    Args:
        formula: Molecular formula string.
        
    Returns:
        DBE value (float; valid formulas have integer or half-integer DBE ≥ 0).
    """
    counts = parse_formula(formula)
    # Standard valences
    valences = {'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1}
    dbe = 1.0
    for elem, count in counts.items():
        v = valences.get(elem, 0)
        dbe += count * (v - 2) / 2.0
    return dbe


def get_elemental_class(formula: str) -> str:
    """
    Classify formula into elemental class.
    
    Returns one of: 'CHO', 'CHON', 'CHOS', 'CHONS', 'CHOF', 'CHONSP', 'other'
    """
    counts = parse_formula(formula)
    has = {e: counts.get(e, 0) > 0 for e in ELEMENTS}
    
    if has['F'] and has['C'] and has['H']:
        return 'CHOF'
    if has['P'] and has['N'] and has['S']:
        return 'CHONSP'
    if has['N'] and has['S']:
        return 'CHONS'
    if has['N'] and not has['S']:
        return 'CHON'
    if has['S'] and not has['N']:
        return 'CHOS'
    if has['C'] and has['H'] and has['O'] and not has['N'] and not has['S'] and not has['F']:
        return 'CHO'
    return 'other'


def get_dbe_bin(dbe: float) -> int:
    """
    Bin DBE into 4 classes: 0-3, 4-7, 8-12, 13+.
    
    Returns bin index (0, 1, 2, or 3).
    """
    if dbe <= 3:
        return 0
    elif dbe <= 7:
        return 1
    elif dbe <= 12:
        return 2
    else:
        return 3


# ─────────────────────────────────────────────────────────────
# Seven Golden Rules validation
# ─────────────────────────────────────────────────────────────

def check_golden_rules(formula: str) -> bool:
    """
    Apply Seven Golden Rules filters to validate a molecular formula.
    
    Rules:
        1. DBE ≥ 0
        2. H ≤ 2C + N + 2  (hydrogen rule)
        3. O/C ≤ 3
        4. N/C ≤ 2
        5. H/C ≤ 3
        6. Valence check (DBE is integer or half-integer)
        7. Electron parity (sum of valence*count must match)
    
    Args:
        formula: Molecular formula string.
        
    Returns:
        True if formula passes all Golden Rules.
    """
    counts = parse_formula(formula)
    c = counts.get('C', 0)
    h = counts.get('H', 0)
    o = counts.get('O', 0)
    n = counts.get('N', 0)
    s = counts.get('S', 0)
    
    # Rule 1: DBE ≥ 0
    dbe = compute_dbe(formula)
    if dbe < 0:
        return False
    
    # Rule 2: Hydrogen rule
    if h > 2 * c + n + 2:
        return False
    
    # Rule 3-5: Ratio checks (avoid div by zero)
    if c > 0:
        if o / c > 3:
            return False
        if n / c > 2:
            return False
        if h / c > 3:
            return False
    
    # Rule 6: Valence check — DBE must be non-negative integer (for even-electron)
    # Allow half-integers for odd-electron species
    if dbe < -0.5:
        return False
    
    # Rule 7: Electron parity
    # For [M-H]- species: total valence sum must be even
    valences = {'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1}
    total_valence = sum(counts.get(e, 0) * valences.get(e, 0) for e in counts)
    # Even-electron species: total should be even
    # (For now, accept both parities as we may have radical species)
    
    return True


# ─────────────────────────────────────────────────────────────
# Data loading functions
# ─────────────────────────────────────────────────────────────

def load_dom_table(filepath: str) -> pd.DataFrame:
    """
    Load a DOM training/testing CSV table.
    Columns: Origin, m/z exp., Chem. Formula, m/z calc., m/z error (ppm), Ion type
    
    Handles duplicate rows by deduplication.
    
    Returns:
        Cleaned DataFrame with standardized column names.
    """
    df = pd.read_csv(filepath, low_memory=False)
    
    # Standardize column names
    col_map = {
        'Origin': 'origin',
        'm/z exp.': 'mz_exp',
        'Chem. Formula': 'formula',
        'm/z calc.': 'mz_calc',
        'm/z error (ppm)': 'ppm_error',
        'Ion type': 'ion_type',
    }
    df = df.rename(columns=col_map)
    
    # Deduplicate (ver2 files have many repeated rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=['mz_exp', 'formula'], keep='first')
    n_after = len(df)
    if n_before != n_after:
        logger.info(f"  Removed {n_before - n_after} duplicate rows from {filepath}")
    
    # Drop rows with missing formulas
    n_na = df['formula'].isna().sum()
    if n_na > 0:
        logger.warning(f"  Dropping {n_na} rows with NaN formula in {filepath}")
        df = df.dropna(subset=['formula']).reset_index(drop=True)
    
    # Ensure formula column is string type
    df['formula'] = df['formula'].astype(str)
    
    # Compute derived features
    df['mass_exact'] = df['formula'].apply(formula_to_mass)
    df['dbe'] = df['formula'].apply(compute_dbe)
    df['elemental_class'] = df['formula'].apply(get_elemental_class)
    df['dbe_bin'] = df['dbe'].apply(get_dbe_bin)
    
    # Quality check: ppm error
    if 'ppm_error' in df.columns:
        bad_ppm = (df['ppm_error'].abs() > 3.0).sum()
        if bad_ppm > 0:
            logger.warning(f"  {bad_ppm} features have |ppm error| > 3.0 in {filepath}")
    
    return df


def load_peak_list(filepath: str) -> pd.DataFrame:
    """
    Load a raw peak list CSV (m/z + intensity, no formula assignments).
    
    Returns:
        DataFrame with columns: mz_exp, intensity
    """
    df = pd.read_csv(filepath)
    col_map = {
        'm/z Exp.': 'mz_exp',
        'Intensity': 'intensity',
    }
    df = df.rename(columns=col_map)
    
    # Basic cleanup
    df = df.dropna(subset=['mz_exp'])
    df = df.sort_values('mz_exp').reset_index(drop=True)
    
    return df


def load_knn_file(filepath: str) -> pd.DataFrame:
    """
    Load KNN-format tab-separated file with ion mobility features.
    Columns: mz, inv_k0, ccs, formula
    
    Returns:
        Cleaned DataFrame with standardized columns and derived features.
    """
    df = pd.read_csv(filepath, sep='\t')
    df = df.rename(columns={'mz': 'mz_exp'})
    
    # Compute derived features
    df['mass_exact'] = df['formula'].apply(formula_to_mass)
    df['dbe'] = df['formula'].apply(compute_dbe)
    df['elemental_class'] = df['formula'].apply(get_elemental_class)
    df['dbe_bin'] = df['dbe'].apply(get_dbe_bin)
    
    return df


# ─────────────────────────────────────────────────────────────
# Full dataset loading & metadata generation
# ─────────────────────────────────────────────────────────────

def load_all_data(data_root: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load all available datasets and create metadata table.
    
    Args:
        data_root: Path to the dataset/ directory.
        
    Returns:
        Tuple of:
          - Dictionary mapping sample_id → DataFrame
          - metadata DataFrame with sample-level info
    """
    datasets: Dict[str, pd.DataFrame] = {}
    metadata_rows: List[Dict] = []
    
    # ── DOM Training ver2 ──
    train_v2_dir = os.path.join(data_root, 'data2_DOM_KNN', 'DOM_training_set_ver2')
    if os.path.isdir(train_v2_dir):
        for fname in sorted(os.listdir(train_v2_dir)):
            if fname.endswith('.csv'):
                fpath = os.path.join(train_v2_dir, fname)
                sample_id = fname.replace('.csv', '').replace('Table_', '')
                logger.info(f"Loading training (ver2): {sample_id}")
                df = load_dom_table(fpath)
                datasets[sample_id] = df
                metadata_rows.append({
                    'sample_id': sample_id,
                    'source': 'DOM_training_ver2',
                    'matrix_type': _infer_matrix(sample_id),
                    'instrument': 'FT-ICR',
                    'resolution': 'standard',
                    'polarity': 'negative',
                    'num_features': len(df),
                    'has_formula': True,
                })
    
    # ── DOM Training ver3 (21T) ──
    train_v3_dir = os.path.join(data_root, 'data2_DOM_KNN', 'DOM_training_set_ver3')
    if os.path.isdir(train_v3_dir):
        for fname in sorted(os.listdir(train_v3_dir)):
            if fname.endswith('.csv'):
                fpath = os.path.join(train_v3_dir, fname)
                sample_id = fname.replace('.csv', '').replace('Table_', '')
                logger.info(f"Loading training (ver3/21T): {sample_id}")
                df = load_dom_table(fpath)
                datasets[sample_id] = df
                metadata_rows.append({
                    'sample_id': sample_id,
                    'source': 'DOM_training_ver3_21T',
                    'matrix_type': _infer_matrix(sample_id),
                    'instrument': 'FT-ICR_21T',
                    'resolution': '21T',
                    'polarity': 'negative',
                    'num_features': len(df),
                    'has_formula': True,
                })
    
    # ── DOM Testing set ──
    test_dir = os.path.join(data_root, 'data2_DOM_KNN', 'DOM_testing_set')
    if os.path.isdir(test_dir):
        for fname in sorted(os.listdir(test_dir)):
            if fname.endswith('.csv'):
                fpath = os.path.join(test_dir, fname)
                sample_id = fname.replace('.csv', '').replace('Table_', '')
                logger.info(f"Loading testing: {sample_id}")
                df = load_dom_table(fpath)
                datasets[sample_id] = df
                metadata_rows.append({
                    'sample_id': sample_id,
                    'source': 'DOM_testing',
                    'matrix_type': _infer_matrix(sample_id),
                    'instrument': 'FT-ICR',
                    'resolution': 'standard',
                    'polarity': 'negative',
                    'num_features': len(df),
                    'has_formula': True,
                })
    
    # ── Peak lists (unassigned) ──
    peak_dir = os.path.join(data_root, 'data2_DOM_KNN', 'DOM_testing_set_Peaklists')
    if os.path.isdir(peak_dir):
        for sub in sorted(os.listdir(peak_dir)):
            sub_path = os.path.join(peak_dir, sub)
            if os.path.isdir(sub_path):
                for fname in sorted(os.listdir(sub_path)):
                    if fname.endswith('.csv'):
                        fpath = os.path.join(sub_path, fname)
                        sample_id = f"PeakList_{sub}"
                        logger.info(f"Loading peak list: {sample_id}")
                        df = load_peak_list(fpath)
                        datasets[sample_id] = df
                        metadata_rows.append({
                            'sample_id': sample_id,
                            'source': 'DOM_peaklist',
                            'matrix_type': _infer_matrix(sub),
                            'instrument': 'FT-ICR',
                            'resolution': 'standard',
                            'polarity': 'negative',
                            'num_features': len(df),
                            'has_formula': False,
                        })
    
    # ── KNN train/test files ──
    knn_dir = os.path.join(data_root, 'data2_DOM_KNN')
    for fname in ['train.txt', 'test.txt']:
        fpath = os.path.join(knn_dir, fname)
        if os.path.isfile(fpath):
            sample_id = f"KNN_{fname.replace('.txt', '')}"
            logger.info(f"Loading KNN data: {sample_id}")
            df = load_knn_file(fpath)
            datasets[sample_id] = df
            metadata_rows.append({
                'sample_id': sample_id,
                'source': 'KNN_mobility',
                'matrix_type': 'mixed_DOM',
                'instrument': 'TIMS-TOF',
                'resolution': 'standard',
                'polarity': 'negative',
                'num_features': len(df),
                'has_formula': True,
            })
    
    # ── PFAS 21T FT-ICR dataset ──
    try:
        from mosaic_gnn.data.pfas_loader import load_pfas_data, _find_pfas_xlsx
        pfas_xlsx = _find_pfas_xlsx(os.path.dirname(data_root))
        if pfas_xlsx is None:
            pfas_xlsx = _find_pfas_xlsx(data_root)
        if pfas_xlsx is not None:
            pfas_datasets, pfas_meta = load_pfas_data(
                pfas_xlsx, min_confidence='Y', include_surfactants=True)
            for sid, df in pfas_datasets.items():
                datasets[sid] = df
            metadata_rows.extend(pfas_meta.to_dict('records'))
            logger.info(f"Loaded {len(pfas_datasets)} PFAS samples")
        else:
            logger.info("No PFAS Excel file found, skipping PFAS data")
    except Exception as e:
        logger.warning(f"Failed to load PFAS data: {e}")

    # ── PANGAEA .tab files (local) ──
    pangaea_dir = os.path.join(data_root, 'PANGAEA — FT-ICR-MS & LC-FT-ICR-MS')
    if os.path.isdir(pangaea_dir):
        for fname, sid, matrix in [
            ('L14_K4_FT-ICR-MS.tab', 'PANGAEA_Lena_L14_K4', 'arctic_river_DOM'),
            ('L14_19_FT-ICR-MS.tab', 'PANGAEA_Lena_L14_19', 'arctic_river_DOM'),
            ('Martinez-Perez_FT.tab', 'PANGAEA_Martinez_Mediterranean', 'marine_DOM'),
        ]:
            fpath = os.path.join(pangaea_dir, fname)
            if os.path.isfile(fpath):
                try:
                    df = _load_pangaea_tab(fpath)
                    if df is not None and len(df) > 0:
                        datasets[sid] = df
                        metadata_rows.append({
                            'sample_id': sid,
                            'source': 'PANGAEA_local',
                            'matrix_type': matrix,
                            'instrument': 'FT-ICR',
                            'resolution': 'standard',
                            'polarity': 'negative',
                            'num_features': len(df),
                            'has_formula': True,
                        })
                        logger.info(f"Loaded PANGAEA tab: {sid} ({len(df)} features)")
                except Exception as e:
                    logger.warning(f"Failed to load {fname}: {e}")

    # ── Osterholz HE361 North Sea DOM (extracted) ──
    osterholz_path = os.path.join(data_root, 'PANGAEA_extracted', 'Osterholz', 'datasets', 'HE361_DOM.tab')
    if os.path.isfile(osterholz_path):
        try:
            df = _load_pangaea_tab(osterholz_path)
            if df is not None and len(df) > 0:
                datasets['PANGAEA_HE361_NorthSea'] = df
                metadata_rows.append({
                    'sample_id': 'PANGAEA_HE361_NorthSea',
                    'source': 'PANGAEA_local',
                    'matrix_type': 'coastal_DOM',
                    'instrument': 'FT-ICR',
                    'resolution': 'standard',
                    'polarity': 'negative',
                    'num_features': len(df),
                    'has_formula': True,
                })
                logger.info(f"Loaded Osterholz HE361: {len(df)} features")
        except Exception as e:
            logger.warning(f"Failed to load Osterholz: {e}")

    # ── Moye 2025 filtered molecular formulas (extracted) ──
    moye_path = os.path.join(data_root, 'PANGAEA_extracted', 'Moye', 'datasets', 'LC-FT-ICR-MS_filtered_molecular_formulas.tab')
    if os.path.isfile(moye_path):
        try:
            df = _load_pangaea_tab(moye_path, skip_events=['Blank_1-1-54_1_2138.d'])
            if df is not None and len(df) > 0:
                # Split by event into separate samples
                if 'sample_id_raw' in df.columns:
                    for event_name, event_df in df.groupby('sample_id_raw'):
                        sid = f"Moye_{event_name.replace('/', '_').replace(' ', '_')[:30]}"
                        event_df = event_df.drop(columns=['sample_id_raw'], errors='ignore')
                        datasets[sid] = event_df.reset_index(drop=True)
                        metadata_rows.append({
                            'sample_id': sid,
                            'source': 'PANGAEA_Moye',
                            'matrix_type': 'marine_DOM',
                            'instrument': 'LC-FT-ICR',
                            'resolution': 'standard',
                            'polarity': 'negative',
                            'num_features': len(event_df),
                            'has_formula': True,
                        })
                        logger.info(f"Loaded Moye event: {sid} ({len(event_df)} features)")
                else:
                    datasets['Moye_all'] = df
                    metadata_rows.append({
                        'sample_id': 'Moye_all',
                        'source': 'PANGAEA_Moye',
                        'matrix_type': 'marine_DOM',
                        'instrument': 'LC-FT-ICR',
                        'resolution': 'standard',
                        'polarity': 'negative',
                        'num_features': len(df),
                        'has_formula': True,
                    })
        except Exception as e:
            logger.warning(f"Failed to load Moye data: {e}")

    # ── External DOM CSV (from download script) ──
    external_csv = os.path.join(data_root, 'external_DOM', 'all_external_DOM.csv')
    if os.path.isfile(external_csv):
        try:
            ext_df = pd.read_csv(external_csv)
            if 'source' in ext_df.columns:
                for src, src_df in ext_df.groupby('source'):
                    # Further split by sample_id if available
                    if 'sample_id' in src_df.columns:
                        for sid_raw, row_df in src_df.groupby('sample_id'):
                            if len(row_df) < 50:
                                continue
                            sid = f"ext_{src}_{str(sid_raw)[:30]}"
                            clean = pd.DataFrame()
                            clean['mz_exp'] = row_df['mz_exp'].values
                            clean['formula'] = row_df['formula'].values
                            if 'intensity' in row_df.columns:
                                clean['intensity'] = row_df['intensity'].values
                            datasets[sid] = clean
                            mt = row_df['matrix_type'].iloc[0] if 'matrix_type' in row_df.columns else 'unknown'
                            metadata_rows.append({
                                'sample_id': sid,
                                'source': f'external_{src}',
                                'matrix_type': mt,
                                'instrument': 'FT-ICR',
                                'resolution': 'standard',
                                'polarity': 'negative',
                                'num_features': len(clean),
                                'has_formula': True,
                            })
                logger.info(f"Loaded external DOM data: {len(ext_df)} total features")
        except Exception as e:
            logger.warning(f"Failed to load external DOM CSV: {e}")

    metadata = pd.DataFrame(metadata_rows)
    return datasets, metadata


def _infer_matrix(sample_id: str) -> str:
    """Infer matrix type from sample ID."""
    sid = sample_id.lower()
    if 'pfas' in sid or 'afff' in sid:
        return 'PFAS_AFFF'
    elif 'harney' in sid:
        return 'estuary'
    elif 'pantanal' in sid:
        return 'tropical_wet'
    elif 'suwannee' in sid or 'srfa' in sid:
        return 'river_DOM'
    elif 'pahokee' in sid or 'ppfa' in sid:
        return 'river_DOM'
    else:
        return 'unknown'


def _load_pangaea_tab(filepath: str, skip_events: list = None) -> Optional[pd.DataFrame]:
    """
    Load a PANGAEA .tab file, skipping the /* ... */ metadata header.
    Returns DataFrame with standardized columns: mz_exp, formula, [intensity].
    """
    lines = open(filepath, encoding='utf-8', errors='replace').readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '*/':
            data_start = i + 1
            break
    if data_start == 0:
        logger.warning(f"  No PANGAEA header found in {filepath}")
        return None

    header = lines[data_start].strip().split('\t')
    rows = []
    for line in lines[data_start + 1:]:
        if line.strip():
            rows.append(line.strip().split('\t'))
    if not rows:
        return None

    raw = pd.DataFrame(rows, columns=header)

    # Find m/z and Formula columns
    mz_col = [c for c in raw.columns if 'm/z' in c.lower()]
    formula_col = [c for c in raw.columns if c.strip().lower() == 'formula'
                   or 'mol' in c.lower() and 'formula' in c.lower()]
    if not formula_col:
        formula_col = [c for c in raw.columns if 'formula' in c.lower()]

    if not mz_col or not formula_col:
        logger.warning(f"  PANGAEA tab missing m/z or Formula: cols={raw.columns.tolist()[:10]}")
        return None

    df = pd.DataFrame()
    df['mz_exp'] = pd.to_numeric(raw[mz_col[0]], errors='coerce')
    df['formula'] = raw[formula_col[0]].astype(str).str.strip()

    # Intensity (optional)
    int_cols = [c for c in raw.columns if any(k in c.lower() for k in ['peak int', 'intensity', 'peak_int'])]
    if int_cols:
        df['intensity'] = pd.to_numeric(raw[int_cols[0]], errors='coerce').fillna(1.0)

    # Event/sample ID (for splitting)
    event_cols = [c for c in raw.columns if c.strip().lower() in ('event', 'sample id', 'sample_id')]
    if event_cols:
        df['sample_id_raw'] = raw[event_cols[0]].astype(str).str.strip()
        if skip_events:
            df = df[~df['sample_id_raw'].isin(skip_events)]

    # Drop NaN m/z or empty formulas
    df = df.dropna(subset=['mz_exp']).reset_index(drop=True)
    df = df[df['formula'].str.len() > 0].reset_index(drop=True)
    df = df[df['formula'] != 'nan'].reset_index(drop=True)

    # Filter: formula must start with C (valid molecular formula)
    df = df[df['formula'].str.startswith('C')].reset_index(drop=True)

    # Compute derived features
    df['mass_exact'] = df['formula'].apply(formula_to_mass)
    df['dbe'] = df['formula'].apply(compute_dbe)
    df['elemental_class'] = df['formula'].apply(get_elemental_class)
    df['dbe_bin'] = df['dbe'].apply(get_dbe_bin)

    return df if len(df) > 0 else None


def chunk_large_sample(
    df: pd.DataFrame,
    sample_id_prefix: str,
    chunk_size: int = 3000,
    min_chunk_size: int = 200,
) -> Dict[str, pd.DataFrame]:
    """
    Split a large DataFrame into smaller chunks sorted by m/z.
    Returns dict of {chunk_sample_id: chunk_df}.
    """
    if len(df) <= chunk_size:
        return {sample_id_prefix: df}

    df_sorted = df.sort_values('mz_exp').reset_index(drop=True)
    chunks = {}
    n_chunks = (len(df_sorted) + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df_sorted))
        chunk_df = df_sorted.iloc[start:end].reset_index(drop=True)
        if len(chunk_df) >= min_chunk_size:
            chunk_id = f"{sample_id_prefix}_chunk{i:03d}"
            chunks[chunk_id] = chunk_df

    return chunks


# ─────────────────────────────────────────────────────────────
# Blank subtraction (noise removal)
# ─────────────────────────────────────────────────────────────

def blank_subtraction(
    sample_df: pd.DataFrame,
    blank_df: Optional[pd.DataFrame] = None,
    sigma_multiplier: float = 3.0,
    intensity_col: str = 'intensity',
) -> pd.DataFrame:
    """
    Remove chemical noise by subtracting blank signal.
    
    Threshold = blank_mean + sigma_multiplier * blank_std
    Features below this threshold are removed.
    
    If no blank is provided, applies a simple low-intensity filter
    (removes features below 0.1% of max intensity).
    """
    if blank_df is not None and intensity_col in blank_df.columns:
        threshold = blank_df[intensity_col].mean() + sigma_multiplier * blank_df[intensity_col].std()
    else:
        # Default: remove bottom 0.1% noise floor
        if intensity_col in sample_df.columns:
            threshold = sample_df[intensity_col].max() * 0.001
        else:
            return sample_df
    
    mask = sample_df[intensity_col] >= threshold
    n_removed = (~mask).sum()
    logger.info(f"  Blank subtraction: removed {n_removed}/{len(sample_df)} features below threshold {threshold:.4e}")
    return sample_df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Quality control
# ─────────────────────────────────────────────────────────────

def quality_control(
    df: pd.DataFrame,
    max_ppm: float = 3.0,
    min_features: int = 100,
) -> Dict[str, bool]:
    """
    Run quality control checks on a sample DataFrame.
    
    Returns dictionary of QC results.
    """
    qc = {}
    
    # Check ppm calibration
    if 'ppm_error' in df.columns:
        rmse = np.sqrt((df['ppm_error'] ** 2).mean())
        qc['calibration_rmse_ppm'] = rmse
        qc['calibration_ok'] = rmse < 0.5
    
    # Check feature count
    qc['num_features'] = len(df)
    qc['features_ok'] = len(df) >= min_features
    
    # Check for valid formulas
    if 'formula' in df.columns:
        valid = df['formula'].apply(check_golden_rules)
        qc['golden_rules_pass_rate'] = valid.mean()
        qc['golden_rules_ok'] = valid.mean() > 0.9
    
    return qc


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Run data loading on available dataset
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset')
    datasets, metadata = load_all_data(data_root)
    
    print(f"\n{'='*60}")
    print(f"MOSAIC-GNN Data Pipeline — P1-T1 Summary")
    print(f"{'='*60}")
    print(f"\nLoaded {len(datasets)} samples:")
    print(metadata.to_string(index=False))
    
    # Run QC on each labeled sample
    print(f"\n{'='*60}")
    print(f"Quality Control Results")
    print(f"{'='*60}")
    for sid, df in datasets.items():
        qc = quality_control(df)
        print(f"\n  {sid}:")
        for k, v in qc.items():
            print(f"    {k}: {v}")
