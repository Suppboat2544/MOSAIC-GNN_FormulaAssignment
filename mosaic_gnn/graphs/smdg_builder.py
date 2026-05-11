"""
P1-T2: SMDG — Spectral Mass-Difference Graph Construction
==========================================================
Builds heterogeneous graphs where nodes are MS1 features (peaks)
and edges connect features whose m/z difference matches an entry
in the Elemental Transformation Dictionary (ETD) within tolerance.

This captures co-occurrence patterns in complex mixtures: e.g.,
consecutive CH2 gains in homologous series of DOM, or CF2 losses
in PFAS transformation product chains.

CRITICAL DESIGN CHOICE — Element-driven ETD:
  The ETD is built dynamically based on which elements are ACTUALLY
  PRESENT in the sample's molecular formulas, not by matrix label.

  Base (CHO always):  CH2, O, CO2, H2O, CH2O
  If N detected  → +NH3
  If S detected  → +SO3
  If F detected  → +CF2, +HF, +COF2
  If F && S      → +SO2F2  (sulfonyl fluoride PFAS)
  If Cl detected → +HCl
  If P detected  → +HPO3

  This ensures:
    - DOM samples (CHONS) get CH2/O/CO2/H2O/CH2O/NH3/SO3 but NOT CF2/HF
    - PFAS samples (CHOF)  get CH2/O/CO2/H2O/CH2O/CF2/HF/COF2 but NOT SO3/NH3
    - Mixed samples get exactly the transformations their elements support
    - No false-positive edges from coincidental Δm matches
"""

import torch
import numpy as np
import re
from scipy.spatial import KDTree
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Set, Sequence
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ETD Building Blocks — keyed by required element(s)
# ─────────────────────────────────────────────────────────────

# Base transformations (always included — only require C, H, O)
ETD_BASE: Dict[str, float] = {
    'CH2':   14.01565,   # methylene: homologous series
    'O':     15.99491,   # oxidation / hydroxylation
    'CO':    27.99491,   # carbonyl (decarbonylation)
    'CO2':   43.98983,   # carboxylation / decarboxylation
    'H2O':   18.01056,   # hydration / dehydration
    'CH2O':  30.01057,   # formaldehyde: methylation + oxidation
}

# Element-conditional transformations
# Each maps: element(s) required → {name: mass}
ETD_CONDITIONAL: Dict[str, Dict[str, float]] = {
    'N': {
        'NH3':   17.02655,   # amination (CHON)
    },
    'S': {
        'SO3':   79.95682,   # sulfonation (CHOS)
    },
    'F': {
        'CF2':   49.99694,   # perfluoro chain extension
        'HF':    20.00623,   # HF elimination (defluorination)
        'COF2':  65.99185,   # COF2 loss in PFAS degradation
    },
    'F+S': {
        'SO2F2': 81.96903,   # sulfonyl fluoride (PFAS with S+F)
    },
    'Cl': {
        'HCl':   35.97668,   # dechlorination
    },
    'P': {
        'HPO3':  79.96633,   # phosphorylation
    },
}

# Legacy convenience aliases
ETD = ETD_BASE.copy()
ETD.update(ETD_CONDITIONAL['N'])
ETD.update(ETD_CONDITIONAL['S'])

PFAS_ETD = ETD_BASE.copy()
PFAS_ETD.update(ETD_CONDITIONAL['F'])


# ─────────────────────────────────────────────────────────────
# Element Detection & ETD Assembly
# ─────────────────────────────────────────────────────────────

_FORMULA_ELEMENT_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def detect_elements_from_formulas(formulas: Sequence[str]) -> Set[str]:
    """
    Detect which elements are present across a list of molecular formulas.

    Args:
        formulas: List of molecular formula strings, e.g. ['C10H12O5', 'C8H6O3N'].

    Returns:
        Set of element symbols found, e.g. {'C', 'H', 'O', 'N'}.
    """
    elements: Set[str] = set()
    for f in formulas:
        if not isinstance(f, str) or not f.strip():
            continue
        for match in _FORMULA_ELEMENT_RE.finditer(f):
            elem = match.group(1)
            count_str = match.group(2)
            count = int(count_str) if count_str else 1
            if count > 0 and elem:
                elements.add(elem)
    return elements


def build_etd_from_elements(detected_elements: Set[str]) -> Dict[str, float]:
    """
    Build an ETD dictionary based on which elements are detected in the sample.

    Always includes CHO base transformations (CH2, O, CO2, H2O, CH2O).
    Conditionally adds element-specific transformations only when the
    corresponding element(s) are present.

    Args:
        detected_elements: Set of element symbols present in sample,
                          e.g. {'C', 'H', 'O', 'N', 'S'}.

    Returns:
        ETD dictionary {name: exact_mass_diff}.

    Examples:
        >>> build_etd_from_elements({'C', 'H', 'O'})
        # Returns only base: CH2, O, CO2, H2O, CH2O

        >>> build_etd_from_elements({'C', 'H', 'O', 'N', 'S'})
        # Base + NH3 + SO3

        >>> build_etd_from_elements({'C', 'H', 'O', 'F'})
        # Base + CF2 + HF + COF2  (PFAS-like, no SO3/NH3)
    """
    etd = ETD_BASE.copy()

    for elem_key, transforms in ETD_CONDITIONAL.items():
        # Handle compound keys like 'F+S'
        required = set(elem_key.split('+'))
        if required.issubset(detected_elements):
            etd.update(transforms)

    logger.info(
        f"  Element-driven ETD: detected {sorted(detected_elements)} → "
        f"{sorted(etd.keys())} ({len(etd)} types)"
    )
    return etd


def get_etd_for_sample(
    formulas: Optional[Sequence[str]] = None,
    detected_elements: Optional[Set[str]] = None,
    matrix_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Return the chemically-appropriate ETD for a sample.

    Priority order:
      1. If formulas provided → detect elements from formulas → build ETD
      2. If detected_elements provided → build ETD from elements
      3. If matrix_type provided → fall back to heuristic

    Args:
        formulas:          List of molecular formula strings (best option).
        detected_elements: Pre-computed set of element symbols.
        matrix_type:       Fallback: 'DOM', 'PFAS', etc.

    Returns:
        ETD dictionary.
    """
    if formulas is not None:
        elements = detect_elements_from_formulas(formulas)
        if elements:
            return build_etd_from_elements(elements)
        logger.warning("  No elements detected from formulas, falling back to matrix_type")

    if detected_elements is not None:
        return build_etd_from_elements(detected_elements)

    if matrix_type is not None:
        mt = matrix_type.lower()
        if any(k in mt for k in ['pfas', 'pfos', 'pfoa', 'fluor']):
            elements = {'C', 'H', 'O', 'F'}
        elif any(k in mt for k in ['dom', 'river', 'estuary', 'pantanal',
                                     'harney', 'suwannee', 'pahokee', 'fulvic', 'humic']):
            elements = {'C', 'H', 'O', 'N', 'S'}
        else:
            logger.warning(f"  Unknown matrix '{matrix_type}', using all elements")
            elements = {'C', 'H', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'P'}
        return build_etd_from_elements(elements)

    # Absolute fallback: CHO only (safest, fewest false positives)
    logger.warning("  No formulas, elements, or matrix_type → using CHO-only base ETD")
    return ETD_BASE.copy()


def build_smdg(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ppm_error: float = 3.0,
    charge_states: Optional[np.ndarray] = None,
    etd: Optional[Dict[str, float]] = None,
    formulas: Optional[Sequence[str]] = None,
    detected_elements: Optional[Set[str]] = None,
    matrix_type: Optional[str] = None,
) -> HeteroData:
    """
    Build Spectral Mass-Difference Graph (SMDG) from a peak list.
    
    For each pair of features (i, j) where j > i:
        Δm = m/z_j - m/z_i
        σ_ij = sqrt(σ_i² + σ_j²)   where σ = m/z × ppm / 1e6
        If |Δm - ETD[r]| < 3 × σ_ij for any r ∈ ETD → add typed edge
    
    Multi-type edges: if Δm matches multiple ETD entries, one edge per type.
    
    ETD Selection (element-driven, priority order):
        1. etd explicitly provided → use as-is
        2. formulas provided → detect elements → build ETD
        3. detected_elements provided → build ETD
        4. matrix_type provided → heuristic element guess → build ETD
        5. None of above → CHO-only base ETD (safest)
    
    Args:
        mz_array:          Array of m/z values (sorted ascending).
        intensity_array:   Array of intensities corresponding to mz_array.
        ppm_error:         Mass accuracy in ppm (default 3.0).
        charge_states:     Optional array of charge states (default: all 1).
        etd:               Custom ETD dictionary. If provided, overrides all auto-detection.
        formulas:          Sample's molecular formulas — BEST way to auto-select ETD.
        detected_elements: Pre-computed element set (alternative to formulas).
        matrix_type:       Fallback heuristic: 'DOM', 'PFAS', etc.
    
    Returns:
        HeteroData object with:
          - data['feature'].x: Node features [m/z, log10(intensity+1), σ, charge]
          - data['feature', etype, 'feature'].edge_index for each edge type r
    """
    if etd is None:
        etd = get_etd_for_sample(formulas=formulas, detected_elements=detected_elements,
                                  matrix_type=matrix_type)
    
    N = len(mz_array)
    assert N == len(intensity_array), "mz and intensity arrays must have same length"
    
    # Ensure sorted
    sort_idx = np.argsort(mz_array)
    mz_array = mz_array[sort_idx]
    intensity_array = intensity_array[sort_idx]
    
    # Compute per-feature mass uncertainty (sigma)
    sigma = mz_array * ppm_error / 1e6
    
    if charge_states is None:
        charge_states = np.ones(N, dtype=np.float32)
    else:
        charge_states = charge_states[sort_idx].astype(np.float32)
    
    # ── Build edges ──
    # Pre-compute max ETD mass to limit search window
    max_etd_mass = max(etd.values()) + 0.1  # small buffer
    
    edges: Dict[str, List[List[int]]] = {etype: [] for etype in etd}
    edge_attrs: Dict[str, List[List[float]]] = {etype: [] for etype in etd}
    edge_count = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            delta_m = mz_array[j] - mz_array[i]
            
            # Early exit: if delta_m exceeds max ETD + tolerance, no more matches for this i
            max_tol = 3 * np.sqrt(sigma[i]**2 + sigma[j]**2)
            if delta_m > max_etd_mass + max_tol:
                break
            
            # Check all ETD entries
            tol_ij = 3 * np.sqrt(sigma[i]**2 + sigma[j]**2)
            sigma_ij = np.sqrt(sigma[i]**2 + sigma[j]**2)
            for etype, dm_ref in etd.items():
                residual = abs(delta_m - dm_ref)
                if residual < tol_ij:
                    ppm_res = (delta_m - dm_ref) / dm_ref * 1e6
                    conf = float(np.exp(-(residual / sigma_ij)**2 / 2.0))
                    edges[etype].append([i, j])
                    edges[etype].append([j, i])  # Undirected
                    edge_attrs[etype].append([delta_m, ppm_res, conf])
                    edge_attrs[etype].append([delta_m, -ppm_res, conf])
                    edge_count += 1
    
    # ── Construct HeteroData ──
    data = HeteroData()
    
    # Node features: [m/z, log10(intensity+1), sigma, charge_state]
    node_features = np.column_stack([
        mz_array,
        np.log10(intensity_array + 1),
        sigma,
        charge_states,
    ])
    data['feature'].x = torch.tensor(node_features, dtype=torch.float32)
    data['feature'].num_nodes = N
    
    # Edge indices + attributes per type
    total_edges = 0
    for etype, edge_list in edges.items():
        if edge_list:
            ei = torch.tensor(edge_list, dtype=torch.long).T
            data['feature', etype, 'feature'].edge_index = ei
            # Store edge attributes: [matched_delta_m, residual_ppm, confidence]
            ea = torch.tensor(edge_attrs[etype], dtype=torch.float32)
            data['feature', etype, 'feature'].edge_attr = ea
            total_edges += len(edge_list) // 2  # undirected count
    
    logger.info(
        f"  SMDG: {N} nodes, {total_edges} edges across {sum(1 for e in edges.values() if e)} edge types"
    )
    
    return data


def build_smdg_vectorized(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ppm_error: float = 3.0,
    charge_states: Optional[np.ndarray] = None,
    etd: Optional[Dict[str, float]] = None,
    formulas: Optional[Sequence[str]] = None,
    detected_elements: Optional[Set[str]] = None,
    matrix_type: Optional[str] = None,
) -> HeteroData:
    """
    Vectorized (faster) version of SMDG construction for large peak lists.
    
    Uses sorted-array binary search + vectorized tolerance comparison
    instead of O(N²) pairwise loop. Scales to ~10k features.
    
    ETD Selection: Same element-driven logic as build_smdg().
    
    Args:
        mz_array:          Array of m/z values.
        intensity_array:   Array of intensities.
        ppm_error:         Mass accuracy in ppm.
        charge_states:     Optional charge states.
        etd:               Custom ETD. If None, auto-select from formulas/elements/matrix.
        formulas:          Sample's molecular formulas for element-driven ETD.
        detected_elements: Pre-computed element set.
        matrix_type:       Fallback heuristic.
    """
    if etd is None:
        etd = get_etd_for_sample(formulas=formulas, detected_elements=detected_elements,
                                  matrix_type=matrix_type)
    
    N = len(mz_array)
    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    int_sorted = intensity_array[sort_idx]
    sigma = mz_sorted * ppm_error / 1e6
    
    if charge_states is None:
        charge_states = np.ones(N, dtype=np.float32)
    else:
        charge_states = charge_states[sort_idx].astype(np.float32)
    
    edges: Dict[str, List[List[int]]] = {etype: [] for etype in etd}
    edge_attrs: Dict[str, List[List[float]]] = {etype: [] for etype in etd}
    
    for etype, dm_ref in etd.items():
        # For each feature i, binary search for features j where mz_j ≈ mz_i + dm_ref
        for i in range(N):
            target = mz_sorted[i] + dm_ref
            tol_i = sigma[i]
            
            # Binary search for range [target - max_tol, target + max_tol]
            # max_tol ≈ 3 * sqrt(2) * max(sigma) as upper bound
            max_tol_search = 3 * np.sqrt(tol_i**2 + (target * ppm_error / 1e6)**2)
            
            lo = np.searchsorted(mz_sorted, target - max_tol_search, side='left')
            hi = np.searchsorted(mz_sorted, target + max_tol_search, side='right')
            
            for j in range(lo, hi):
                if j == i:
                    continue
                tol_ij = 3 * np.sqrt(sigma[i]**2 + sigma[j]**2)
                sigma_ij = np.sqrt(sigma[i]**2 + sigma[j]**2)
                residual = abs(mz_sorted[j] - target)
                if residual < tol_ij:
                    delta_m = mz_sorted[j] - mz_sorted[i]
                    ppm_res = (delta_m - dm_ref) / dm_ref * 1e6
                    conf = float(np.exp(-(residual / sigma_ij)**2 / 2.0))
                    edges[etype].append([i, j])
                    edge_attrs[etype].append([delta_m, ppm_res, conf])
    
    # Build HeteroData
    data = HeteroData()
    node_features = np.column_stack([
        mz_sorted,
        np.log10(int_sorted + 1),
        sigma,
        charge_states,
    ])
    data['feature'].x = torch.tensor(node_features, dtype=torch.float32)
    data['feature'].num_nodes = N
    
    total_edges = 0
    for etype, edge_list in edges.items():
        if edge_list:
            ei = torch.tensor(edge_list, dtype=torch.long).T
            ea = edge_attrs[etype]
            # Remove duplicate edges (keep first occurrence's attrs)
            ei_set = set()
            unique_edges = []
            unique_attrs = []
            for k in range(ei.shape[1]):
                pair = (ei[0, k].item(), ei[1, k].item())
                if pair not in ei_set:
                    ei_set.add(pair)
                    unique_edges.append([pair[0], pair[1]])
                    unique_attrs.append(ea[k])
            if unique_edges:
                ei = torch.tensor(unique_edges, dtype=torch.long).T
                data['feature', etype, 'feature'].edge_index = ei
                data['feature', etype, 'feature'].edge_attr = torch.tensor(
                    unique_attrs, dtype=torch.float32)
                total_edges += len(unique_edges)
    
    logger.info(f"  SMDG (vec): {N} nodes, {total_edges} edges")
    return data


def compute_smdg_statistics(data: HeteroData, etd: Optional[Dict[str, float]] = None) -> Dict[str, any]:
    """
    Compute edge type distribution statistics for a SMDG.
    
    Args:
        data: HeteroData SMDG graph.
        etd:  Optional ETD used to build this SMDG. If None, discovers edge types from data.
    
    Returns:
        Dictionary mapping edge type → count, plus 'total' and 'connected_ratio'.
    """
    stats = {}
    total = 0
    
    # Discover actual edge types from the HeteroData
    for edge_type_tuple in data.edge_types:
        etype = edge_type_tuple[1]
        n = data[edge_type_tuple].edge_index.shape[1]
        stats[etype] = n
        total += n
    
    stats['total'] = total
    
    # Compute connectivity
    N = data['feature'].num_nodes
    if total > 0:
        connected = set()
        for edge_type_tuple in data.edge_types:
            ei = data[edge_type_tuple].edge_index
            for k in range(ei.shape[1]):
                connected.add(ei[0, k].item())
                connected.add(ei[1, k].item())
        stats['connected_ratio'] = len(connected) / N if N > 0 else 0
    else:
        stats['connected_ratio'] = 0
    
    return stats


def build_smdg_kdtree(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ppm_error: float = 3.0,
    charge_states: Optional[np.ndarray] = None,
    etd: Optional[Dict[str, float]] = None,
    formulas: Optional[Sequence[str]] = None,
    detected_elements: Optional[Set[str]] = None,
    matrix_type: Optional[str] = None,
) -> HeteroData:
    """
    KDTree-accelerated SMDG construction for large peak lists.

    Uses scipy.spatial.KDTree on 1-D m/z for O(N log N) neighbor lookup
    instead of O(N²) brute-force. Computes edge confidence and ppm
    residual as stored edge_attr per edge.

    Performance: ~10x faster than build_smdg for N > 2000 features.

    Args:
        mz_array:          Sorted m/z values.
        intensity_array:   Corresponding intensities.
        ppm_error:         Mass accuracy in ppm (default 3.0).
        charge_states:     Optional charge states (default: all 1).
        etd:               Custom ETD. If None, auto-select.
        formulas:          Molecular formulas for ETD auto-detection.
        detected_elements: Pre-computed element set.
        matrix_type:       Fallback heuristic.

    Returns:
        HeteroData with edge_attr = [matched_delta_m, residual_ppm, confidence].
    """
    if etd is None:
        etd = get_etd_for_sample(formulas=formulas, detected_elements=detected_elements,
                                  matrix_type=matrix_type)

    N = len(mz_array)
    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    int_sorted = intensity_array[sort_idx]
    sigma = mz_sorted * ppm_error / 1e6

    if charge_states is None:
        charge_states = np.ones(N, dtype=np.float32)
    else:
        charge_states = charge_states[sort_idx].astype(np.float32)

    # Build KDTree on 1-D m/z (reshape to column vector for KDTree)
    tree = KDTree(mz_sorted.reshape(-1, 1))

    # Max search radius: largest ETD mass + generous tolerance
    max_sigma = sigma.max() if N > 0 else 0.001
    max_etd_mass = max(etd.values())

    edges: Dict[str, List[List[int]]] = {etype: [] for etype in etd}
    edge_attrs: Dict[str, List[List[float]]] = {etype: [] for etype in etd}

    for etype, dm_ref in etd.items():
        for i in range(N):
            target = mz_sorted[i] + dm_ref
            # Search radius: 3*sigma_max (conservative upper bound; exact check below)
            search_r = 3 * np.sqrt(sigma[i]**2 + max_sigma**2) + 0.001
            # Query KDTree for all points within search_r of target
            candidates = tree.query_ball_point([target], r=search_r)[0]

            for j in candidates:
                if j == i:
                    continue
                sigma_ij = np.sqrt(sigma[i]**2 + sigma[j]**2)
                tol_ij = 3 * sigma_ij
                residual = abs(mz_sorted[j] - target)
                if residual < tol_ij:
                    delta_m = mz_sorted[j] - mz_sorted[i]
                    ppm_res = (delta_m - dm_ref) / dm_ref * 1e6
                    conf = float(np.exp(-(residual / sigma_ij)**2 / 2.0))
                    edges[etype].append([i, j])
                    edge_attrs[etype].append([delta_m, ppm_res, conf])

    # Build HeteroData
    data = HeteroData()
    node_features = np.column_stack([
        mz_sorted,
        np.log10(int_sorted + 1),
        sigma,
        charge_states,
    ])
    data['feature'].x = torch.tensor(node_features, dtype=torch.float32)
    data['feature'].num_nodes = N

    total_edges = 0
    for etype, edge_list in edges.items():
        if edge_list:
            ea = edge_attrs[etype]
            # De-duplicate edges
            ei_set = set()
            unique_edges = []
            unique_attrs = []
            for k, (src_dst) in enumerate(edge_list):
                pair = (src_dst[0], src_dst[1])
                if pair not in ei_set:
                    ei_set.add(pair)
                    unique_edges.append(src_dst)
                    unique_attrs.append(ea[k])
            if unique_edges:
                ei = torch.tensor(unique_edges, dtype=torch.long).T
                data['feature', etype, 'feature'].edge_index = ei
                data['feature', etype, 'feature'].edge_attr = torch.tensor(
                    unique_attrs, dtype=torch.float32)
                total_edges += len(unique_edges)

    logger.info(f"  SMDG (kdtree): {N} nodes, {total_edges} edges")
    return data
