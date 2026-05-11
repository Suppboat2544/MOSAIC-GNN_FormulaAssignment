"""
P1-T3: G_ISO — Isotope Envelope Graph Construction
===================================================
Builds isotopologue sub-graphs for each monoisotopic feature.

For each monoisotopic peak at m/z_0:
  - Search for M+1 at m/z_0 + 1.003355  (¹³C)
  - Search for M+2 at m/z_0 + 2.004969  (¹³C×2)
  - Search for M+2 at m/z_0 + 1.995797  (³⁴S)
  - Search for M+2 at m/z_0 + 1.997050  (³⁷Cl)
  - Search for M+2 at m/z_0 + 1.997924  (⁸¹Br)

Edges are typed by isotope species (e_13C, e_34S, e_37Cl, e_81Br).
Node features include observed vs theoretical intensity ratios.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
from scipy.special import comb
import logging

logger = logging.getLogger(__name__)

# Natural isotope abundances (IUPAC)
ISOTOPE_ABUNDANCES: Dict[str, float] = {
    'C':  0.01103,   # ¹³C/¹²C
    'S':  0.04250,   # ³⁴S/³²S
    'Cl': 0.24229,   # ³⁷Cl/³⁵Cl
    'Br': 0.49314,   # ⁸¹Br/⁷⁹Br
}

# Isotope mass shifts (Da)
ISOTOPE_SHIFTS: Dict[str, float] = {
    '13C_M1':  1.003355,   # M+1 from ¹³C
    '13C_M2':  2.004969,   # M+2 from ¹³C×2
    '34S_M2':  1.995797,   # M+2 from ³⁴S
    '37Cl_M2': 1.997050,   # M+2 from ³⁷Cl
    '81Br_M2': 1.997924,   # M+2 from ⁸¹Br
}

# Edge type mapping
ISO_EDGE_TYPES: Dict[str, int] = {
    '13C_M1':  0,
    '13C_M2':  1,
    '34S_M2':  2,
    '37Cl_M2': 3,
    '81Br_M2': 4,
}


def roepstorff_layer(atom_counts: Dict[str, int], k: int) -> float:
    """
    Differentiable isotope intensity prediction using the Roepstorff model.
    
    Computes theoretical I(M+k)/I(M) for a given molecular formula.
    Based on binomial expansion of natural isotope abundances.
    
    For M+1 (k=1):
        I(M+1)/I(M) ≈ n_C × 0.01103 + n_S × 0.00760 + n_N × 0.00369 + ...
    
    For M+2 (k=2):
        Contributions from ¹³C×2, ³⁴S, ³⁷Cl, ⁸¹Br
    
    Args:
        atom_counts: Dictionary of element → count.
        k: Isotopologue order (1 for M+1, 2 for M+2).
    
    Returns:
        Theoretical intensity ratio I(M+k)/I(M).
    """
    if k == 1:
        # M+1 is dominated by ¹³C contribution
        n_c = atom_counts.get('C', 0)
        n_n = atom_counts.get('N', 0)
        n_s = atom_counts.get('S', 0)
        n_h = atom_counts.get('H', 0)
        
        # ¹³C contribution (dominant)
        ratio_13c = n_c * 0.01103
        # ²H (deuterium): very small, 0.0115%
        ratio_2h = n_h * 0.000115
        # ¹⁵N: 0.368%
        ratio_15n = n_n * 0.00368
        # ³³S: 0.760%
        ratio_33s = n_s * 0.00760
        
        return ratio_13c + ratio_2h + ratio_15n + ratio_33s
    
    elif k == 2:
        # M+2 has multiple contributions
        n_c = atom_counts.get('C', 0)
        n_s = atom_counts.get('S', 0)
        n_cl = atom_counts.get('Cl', 0)
        n_br = atom_counts.get('Br', 0)
        
        # ¹³C×2: C(n_c, 2) × (0.01103)²
        ratio_13c2 = comb(n_c, 2, exact=True) * (0.01103**2)
        # ³⁴S: n_s × 0.04250
        ratio_34s = n_s * 0.04250
        # ³⁷Cl: n_cl × 0.24229
        ratio_37cl = n_cl * 0.24229
        # ⁸¹Br: n_br × 0.49314
        ratio_81br = n_br * 0.49314
        # ¹⁸O: n_o × 0.00205
        n_o = atom_counts.get('O', 0)
        ratio_18o = n_o * 0.00205
        
        return ratio_13c2 + ratio_34s + ratio_37cl + ratio_81br + ratio_18o
    
    else:
        # Higher order — simplified
        return 0.0


class RoepstorffPhysicsLayer(torch.nn.Module):
    """
    Differentiable PyTorch implementation of the Roepstorff isotope model.
    
    Given predicted atom counts (as continuous values), computes theoretical
    isotope ratios that can be backpropagated through.
    """
    
    def __init__(self):
        super().__init__()
        # Register isotope abundances as buffers (non-trainable)
        self.register_buffer('p_13C', torch.tensor(0.01103, dtype=torch.float32))
        self.register_buffer('p_2H', torch.tensor(0.000115, dtype=torch.float32))
        self.register_buffer('p_15N', torch.tensor(0.00368, dtype=torch.float32))
        self.register_buffer('p_33S', torch.tensor(0.00760, dtype=torch.float32))
        self.register_buffer('p_34S', torch.tensor(0.04250, dtype=torch.float32))
        self.register_buffer('p_37Cl', torch.tensor(0.24229, dtype=torch.float32))
        self.register_buffer('p_81Br', torch.tensor(0.49314, dtype=torch.float32))
        self.register_buffer('p_18O', torch.tensor(0.00205, dtype=torch.float32))
    
    def forward(self, atom_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute theoretical M+1 and M+2 ratios.
        
        Args:
            atom_counts: Tensor of shape (batch, 9) — [C, H, O, N, S, P, F, Cl, Br]
        
        Returns:
            Tuple of (ratio_M1, ratio_M2), each of shape (batch,)
        """
        n_c  = atom_counts[:, 0]  # Carbon
        n_h  = atom_counts[:, 1]  # Hydrogen
        n_o  = atom_counts[:, 2]  # Oxygen
        n_n  = atom_counts[:, 3]  # Nitrogen
        n_s  = atom_counts[:, 4]  # Sulfur
        n_cl = atom_counts[:, 7]  # Chlorine
        n_br = atom_counts[:, 8]  # Bromine
        
        # M+1 ratio
        ratio_m1 = (n_c * self.p_13C + n_h * self.p_2H +
                     n_n * self.p_15N + n_s * self.p_33S)
        
        # M+2 ratio
        ratio_13c2 = 0.5 * n_c * (n_c - 1) * (self.p_13C ** 2)
        ratio_m2 = (ratio_13c2 +
                     n_s * self.p_34S +
                     n_cl * self.p_37Cl +
                     n_br * self.p_81Br +
                     n_o * self.p_18O)
        
        return ratio_m1, ratio_m2


def build_iso_graph(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ppm_error: float = 3.0,
) -> Tuple[List[Data], Dict[str, float]]:
    """
    Build Isotope Envelope Graphs (G_ISO) for all features.
    
    For each peak, search for M+1 and M+2 isotopologues.
    Build a sub-graph per monoisotopic feature.
    
    Args:
        mz_array:        Sorted m/z values.
        intensity_array: Corresponding intensities.
        ppm_error:       Mass accuracy in ppm.
    
    Returns:
        Tuple of:
          - List of PyG Data objects (one per monoisotopic feature with isotopologues found)
          - Dictionary of statistics: num_with_M1, num_with_M2, etc.
    """
    N = len(mz_array)
    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    int_sorted = intensity_array[sort_idx]
    
    iso_graphs: List[Data] = []
    stats = {
        'total_features': N,
        'features_with_M1': 0,
        'features_with_M2': 0,
        'features_with_isotopologues': 0,
    }
    
    # Store per-feature isotope data
    # Each iso_graph: nodes = [monoisotopic, M+1, M+2_13C, M+2_34S, M+2_37Cl, M+2_81Br]
    iso_feature_data = []
    
    for i in range(N):
        mz_mono = mz_sorted[i]
        int_mono = int_sorted[i]
        sigma_mono = mz_mono * ppm_error / 1e6
        
        # Initialize: node 0 = monoisotopic
        nodes = [{'mz': mz_mono, 'intensity': int_mono, 'type': 'mono'}]
        edge_src = []
        edge_dst = []
        edge_types = []
        
        found_any = False
        
        # Search for M+1 (¹³C)
        target_m1 = mz_mono + ISOTOPE_SHIFTS['13C_M1']
        tol_m1 = 3 * np.sqrt(sigma_mono**2 + (target_m1 * ppm_error / 1e6)**2)
        lo = np.searchsorted(mz_sorted, target_m1 - tol_m1, side='left')
        hi = np.searchsorted(mz_sorted, target_m1 + tol_m1, side='right')
        
        best_m1_idx = None
        best_m1_err = float('inf')
        for j in range(lo, hi):
            if j == i:
                continue
            err = abs(mz_sorted[j] - target_m1)
            if err < best_m1_err:
                best_m1_err = err
                best_m1_idx = j
        
        if best_m1_idx is not None:
            found_any = True
            stats['features_with_M1'] += 1
            node_id = len(nodes)
            nodes.append({
                'mz': mz_sorted[best_m1_idx],
                'intensity': int_sorted[best_m1_idx],
                'type': '13C_M1',
                'observed_ratio': int_sorted[best_m1_idx] / (int_mono + 1e-10),
            })
            edge_src.extend([0, node_id])
            edge_dst.extend([node_id, 0])
            edge_types.extend([ISO_EDGE_TYPES['13C_M1'], ISO_EDGE_TYPES['13C_M1']])
        
        # Search for M+2 peaks
        found_m2 = False
        for shift_name in ['13C_M2', '34S_M2', '37Cl_M2', '81Br_M2']:
            target = mz_mono + ISOTOPE_SHIFTS[shift_name]
            tol = 3 * np.sqrt(sigma_mono**2 + (target * ppm_error / 1e6)**2)
            lo = np.searchsorted(mz_sorted, target - tol, side='left')
            hi = np.searchsorted(mz_sorted, target + tol, side='right')
            
            best_idx = None
            best_err = float('inf')
            for j in range(lo, hi):
                if j == i:
                    continue
                err = abs(mz_sorted[j] - target)
                if err < best_err:
                    best_err = err
                    best_idx = j
            
            if best_idx is not None:
                found_any = True
                found_m2 = True
                node_id = len(nodes)
                nodes.append({
                    'mz': mz_sorted[best_idx],
                    'intensity': int_sorted[best_idx],
                    'type': shift_name,
                    'observed_ratio': int_sorted[best_idx] / (int_mono + 1e-10),
                })
                edge_src.extend([0, node_id])
                edge_dst.extend([node_id, 0])
                edge_types.extend([ISO_EDGE_TYPES[shift_name], ISO_EDGE_TYPES[shift_name]])
        
        if found_m2:
            stats['features_with_M2'] += 1
        
        if found_any:
            stats['features_with_isotopologues'] += 1
        
        # Build PyG Data for this feature's isotope sub-graph
        n_nodes = len(nodes)
        
        # Node features: [m/z, log10(int+1), observed_ratio, is_mono]
        x = torch.zeros(n_nodes, 4, dtype=torch.float32)
        for k, node in enumerate(nodes):
            x[k, 0] = float(node['mz'])
            x[k, 1] = float(np.log10(float(node['intensity']) + 1))
            x[k, 2] = float(node.get('observed_ratio', 1.0))
            x[k, 3] = 1.0 if node['type'] == 'mono' else 0.0
        
        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        
        iso_data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            mono_idx=torch.tensor([i]),       # index in original mz_array
            num_nodes=n_nodes,
        )
        iso_graphs.append(iso_data)
        
        # Store observed ratios for physics validation
        iso_entry = {
            'feature_idx': i,
            'mz': mz_mono,
            'has_M1': best_m1_idx is not None,
            'obs_ratio_M1': nodes[1]['observed_ratio'] if len(nodes) > 1 and nodes[1]['type'] == '13C_M1' else 0.0,
        }
        iso_feature_data.append(iso_entry)
    
    logger.info(
        f"  ISO: {stats['features_with_isotopologues']}/{N} features have isotopologue matches "
        f"(M+1: {stats['features_with_M1']}, M+2: {stats['features_with_M2']})"
    )
    
    return iso_graphs, stats


def compute_isotope_fit_score(
    observed_ratio_m1: float,
    observed_ratio_m2: float,
    atom_counts: Dict[str, int],
) -> float:
    """
    Compute chi-squared isotope fit score.
    
    χ² = Σ (obs - theo)² / theo
    
    Lower is better (good fit → χ² < 0.1).
    """
    theo_m1 = roepstorff_layer(atom_counts, k=1)
    theo_m2 = roepstorff_layer(atom_counts, k=2)
    
    chi_sq = 0.0
    if theo_m1 > 1e-6:
        chi_sq += (observed_ratio_m1 - theo_m1)**2 / theo_m1
    if theo_m2 > 1e-6:
        chi_sq += (observed_ratio_m2 - theo_m2)**2 / theo_m2
    
    return chi_sq
