"""
P1-T4: G_ADC — Adduct Consistency Graph Construction
=====================================================
Builds adduct-typed edges between features that may be
different ion forms (adducts) of the same neutral molecule.

In ESI-MS, a single molecule M can appear as multiple peaks:
  [M+H]+, [M+Na]+, [M+K]+, [M+NH4]+  (positive mode)
  [M-H]-, [M+Cl]-                      (negative mode)

Features connected by adduct edges should yield the same
neutral mass M after subtracting the adduct offset.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Adduct offset dictionary
# Key: adduct name, Value: mass offset from neutral M
ADDUCT_OFFSETS: Dict[str, float] = {
    '[M+H]+':   1.00728,
    '[M+Na]+':  22.98922,
    '[M+K]+':   38.96316,
    '[M+NH4]+': 18.03437,
    '[M-H]-':   -1.00728,
    '[M+Cl]-':  -34.96941,
}

# For same-polarity adduct pairing (mass differences between adducts)
# e.g. [M+Na]+ - [M+H]+ = 21.98194 Da
ADDUCT_PAIRS: Dict[str, Tuple[str, str, float]] = {}


def _precompute_adduct_pairs():
    """Pre-compute mass differences between all adduct pairs of same polarity."""
    pos_adducts = {k: v for k, v in ADDUCT_OFFSETS.items() if '+' in k.split(']')[0][-1:] or k.endswith('+')}
    neg_adducts = {k: v for k, v in ADDUCT_OFFSETS.items() if '-' in k.split(']')[0][-1:] or k.endswith('-')}
    
    # Actually split by polarity more carefully
    pos_adducts = {k: v for k, v in ADDUCT_OFFSETS.items() if k.endswith('+')}
    neg_adducts = {k: v for k, v in ADDUCT_OFFSETS.items() if k.endswith('-')}
    
    pairs = {}
    for name1, off1 in pos_adducts.items():
        for name2, off2 in pos_adducts.items():
            if name1 < name2:  # avoid duplicates
                delta = off2 - off1
                pair_name = f"{name1}_to_{name2}"
                pairs[pair_name] = (name1, name2, delta)
    
    for name1, off1 in neg_adducts.items():
        for name2, off2 in neg_adducts.items():
            if name1 < name2:
                delta = off2 - off1
                pair_name = f"{name1}_to_{name2}"
                pairs[pair_name] = (name1, name2, delta)
    
    return pairs


# Pre-compute on import
ADDUCT_PAIRS = _precompute_adduct_pairs()


def build_adc_graph(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ppm_error: float = 3.0,
    polarity: str = 'negative',
) -> Data:
    """
    Build Adduct Consistency Graph (G_ADC).
    
    For each pair of features, check if their m/z difference matches
    any adduct offset combination. If so, they may be adducts of the
    same neutral molecule.
    
    Args:
        mz_array:        m/z values.
        intensity_array: Corresponding intensities.
        ppm_error:       Mass accuracy in ppm.
        polarity:        'positive' or 'negative' mode.
    
    Returns:
        PyG Data object with adduct edges and neutral mass features.
    """
    N = len(mz_array)
    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    int_sorted = intensity_array[sort_idx]
    sigma = mz_sorted * ppm_error / 1e6
    
    # Select relevant adduct pairs based on polarity
    if polarity == 'negative':
        relevant_adducts = {k: v for k, v in ADDUCT_OFFSETS.items() if k.endswith('-')}
    elif polarity == 'positive':
        relevant_adducts = {k: v for k, v in ADDUCT_OFFSETS.items() if k.endswith('+')}
    else:
        relevant_adducts = ADDUCT_OFFSETS
    
    # Compute all pairwise adduct offset differences
    adduct_deltas: Dict[str, float] = {}
    adduct_names = list(relevant_adducts.keys())
    for i_a, (name1, off1) in enumerate(relevant_adducts.items()):
        for j_a, (name2, off2) in enumerate(relevant_adducts.items()):
            if i_a < j_a:
                delta = abs(off2 - off1)
                pair_name = f"{name1}|{name2}"
                adduct_deltas[pair_name] = delta
    
    # Build edges
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_adduct_type: List[int] = []
    edge_neutral_mass_diff: List[float] = []
    
    adduct_type_map = {name: idx for idx, name in enumerate(adduct_deltas.keys())}
    
    for i in range(N):
        for pair_name, delta in adduct_deltas.items():
            # Search for features at mz_i + delta (± tolerance)
            target = mz_sorted[i] + delta
            tol = 3 * np.sqrt(sigma[i]**2 + (target * ppm_error / 1e6)**2)
            
            lo = np.searchsorted(mz_sorted, target - tol, side='left')
            hi = np.searchsorted(mz_sorted, target + tol, side='right')
            
            for j in range(lo, hi):
                if j <= i:
                    continue
                err = abs(mz_sorted[j] - target)
                tol_ij = 3 * np.sqrt(sigma[i]**2 + sigma[j]**2)
                if err < tol_ij:
                    edge_src.extend([i, j])
                    edge_dst.extend([j, i])
                    etype_idx = adduct_type_map[pair_name]
                    edge_adduct_type.extend([etype_idx, etype_idx])
                    
                    # Compute neutral mass disagreement
                    # For each adduct form, compute what M would be
                    names = pair_name.split('|')
                    off1 = relevant_adducts[names[0]]
                    off2 = relevant_adducts[names[1]]
                    M_i = mz_sorted[i] - off1
                    M_j = mz_sorted[j] - off2
                    mass_diff = abs(M_i - M_j)
                    edge_neutral_mass_diff.extend([mass_diff, mass_diff])
    
    # Build Data object
    node_features = np.column_stack([
        mz_sorted,
        np.log10(int_sorted + 1),
        sigma,
    ])
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(
            list(zip(edge_adduct_type, edge_neutral_mass_diff)),
            dtype=torch.float32
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
    )
    
    # Find adduct families via connected components
    n_edges = edge_index.shape[1] // 2 if edge_index.shape[1] > 0 else 0
    n_families = _count_connected_components(N, edge_index) if n_edges > 0 else N
    
    logger.info(
        f"  ADC: {N} nodes, {n_edges} adduct edges, {n_families} adduct families"
    )
    
    return data


def _count_connected_components(N: int, edge_index: torch.Tensor) -> int:
    """Count connected components using union-find."""
    parent = list(range(N))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for k in range(edge_index.shape[1]):
        i = edge_index[0, k].item()
        j = edge_index[1, k].item()
        union(i, j)
    
    return len(set(find(i) for i in range(N)))
