"""
P1-T5: G_FCL — Formula Candidate Lattice Graph Construction
============================================================
For each feature's accurate mass, enumerate all candidate molecular
formulas within ±3 ppm, filter by Seven Golden Rules, and build a
lattice graph where candidates differing by one unit element are
connected by typed edges.

This is the core "search space" graph that the GIN scorer (Module 3)
will operate on. Each candidate node carries chemical descriptors:
  [DBE, O/C, H/C, N/C, S/C, F/C, ppm_error, mass_exact, valence_ok, electron_parity]
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
from itertools import product
import logging

from ..data.data_pipeline import (
    ATOMIC_MASSES, ELEMENTS, parse_formula, formula_to_mass,
    compute_dbe, check_golden_rules, get_elemental_class, get_dbe_bin,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Bethe-Weizsäcker (BW) Nuclear Mass Defect Features
# ─────────────────────────────────────────────────────────────
# Semi-empirical nuclear mass formula terms for discriminating
# between candidate formulas that have similar exact masses but
# different nuclear binding characteristics.

# Bethe-Weizsäcker coefficients (MeV)
_BW_AV = 15.67      # volume term
_BW_AS = 17.23      # surface term
_BW_AC = 0.714      # Coulomb term
_BW_AA = 93.15      # asymmetry term
# Atomic mass unit in Da
_AMU_TO_MEV = 931.494


def compute_bw_features(counts: Dict[str, int]) -> np.ndarray:
    """
    Compute 4-dim Bethe-Weizsäcker nuclear mass defect features
    for a molecular formula.

    Features:
      [0] total_nucleons (A) — sum of all nucleons
      [1] total_protons (Z) — sum of all protons
      [2] mass_defect — difference between exact mass and A (amu units)
      [3] binding_energy_per_nucleon — semi-empirical BW estimate (MeV/A)

    These features help discriminate isobaric candidates whose exact
    masses agree to <1 ppm but whose nuclear compositions differ.

    Args:
        counts: Element count dictionary, e.g. {'C': 10, 'H': 20, 'O': 5}.

    Returns:
        4-element float32 array: [A, Z, mass_defect, BE/A].
    """
    # Nucleon & proton counts per element
    NUCLEONS = {
        'C': 12, 'H': 1, 'O': 16, 'N': 14, 'S': 32,
        'P': 31, 'F': 19, 'Cl': 35, 'Br': 79, 'I': 127, 'Si': 28,
    }
    PROTONS = {
        'C': 6, 'H': 1, 'O': 8, 'N': 7, 'S': 16,
        'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53, 'Si': 14,
    }

    A = sum(NUCLEONS.get(e, 0) * n for e, n in counts.items())
    Z = sum(PROTONS.get(e, 0) * n for e, n in counts.items())

    if A == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Exact molecular mass from atomic masses
    exact_mass = sum(ATOMIC_MASSES.get(e, 0.0) * n for e, n in counts.items())
    mass_defect = exact_mass - A  # in Da

    # Semi-empirical binding energy (Bethe-Weizsäcker formula)
    A_f = float(A)
    Z_f = float(Z)
    N_f = A_f - Z_f  # neutrons

    BE = (
        _BW_AV * A_f                              # volume
        - _BW_AS * (A_f ** (2.0 / 3.0))           # surface
        - _BW_AC * Z_f * (Z_f - 1) / (A_f ** (1.0 / 3.0))  # Coulomb
        - _BW_AA * (N_f - Z_f) ** 2 / A_f         # asymmetry
    )
    be_per_nucleon = BE / A_f  # MeV/nucleon

    return np.array([
        A_f / 1000.0,              # normalized total nucleons
        Z_f / 500.0,               # normalized total protons
        mass_defect / 10.0,        # normalized mass defect
        be_per_nucleon / 10.0,     # normalized BE/A
    ], dtype=np.float32)


def _formulas_match(f1: str, f2: str) -> bool:
    """Compare two formula strings by element counts (order-independent).

    Handles different element ordering conventions:
      - Mass spectrometry: C8HO2F15 (C,H,O,...,F)
      - Hill notation:     C8HF15O2 (C,H,then alphabetical)
    """
    return parse_formula(f1) == parse_formula(f2)

# Element valences for validation
VALENCES: Dict[str, int] = {
    'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1,
    'I': 1, 'Si': 4,
}

# Default element bounds (from Seven Golden Rules, adapted for each matrix)
DEFAULT_BOUNDS: Dict[str, Tuple[int, int]] = {
    'C':  (1, 100),
    'H':  (0, 200),
    'O':  (0, 50),
    'N':  (0, 10),
    'S':  (0, 5),
    'P':  (0, 5),
    'F':  (0, 30),
    'Cl': (0, 10),
    'Br': (0, 5),
    'I':  (0, 3),
    'Si': (0, 10),
}

# Matrix-specific element priors (which elements to include)
MATRIX_ELEMENT_SETS: Dict[str, List[str]] = {
    'CHO':    ['C', 'H', 'O'],
    'CHON':   ['C', 'H', 'O', 'N'],
    'CHOS':   ['C', 'H', 'O', 'S'],
    'CHONS':  ['C', 'H', 'O', 'N', 'S'],
    'CHOF':   ['C', 'H', 'O', 'F'],
    'CHONSP': ['C', 'H', 'O', 'N', 'S', 'P'],
    'DOM':    ['C', 'H', 'O', 'N', 'S'],
    'PFAS':   ['C', 'H', 'O', 'F'],
    'PFAS_AFFF': ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'I'],
    'metabolomics': ['C', 'H', 'O', 'N', 'S', 'P'],
    'full':   ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si'],
}

# Unit element differences for lattice edges
LATTICE_EDGE_TYPES: List[Tuple[str, int]] = [
    ('C', 1), ('C', -1),
    ('H', 2), ('H', -2),    # H2 for electron parity
    ('O', 1), ('O', -1),
    ('N', 1), ('N', -1),
    ('S', 1), ('S', -1),
]


def enumerate_candidates(
    target_mass: float,
    ppm_tolerance: float = 3.0,
    element_set: str = 'DOM',
    max_candidates: int = 500,
    ion_mode: str = 'negative',
    inject_formula: Optional[str] = None,
) -> List[Dict]:
    """
    Enumerate all valid molecular formula candidates for a given mass.
    
    Uses constrained enumeration with Seven Golden Rules filtering.
    For [M-H]- mode: target neutral mass = observed_mz + 1.00728 Da.
    Optimized: fixes all heavy atoms in nested loops with tight per-level
    bounds, solves for H analytically.
    
    For PFAS_AFFF element sets, uses multi-pass enumeration with
    progressively larger sub-element sets to ensure high-F formulas
    are found before hitting the bail_out limit.
    
    Args:
        target_mass:      Observed m/z value.
        ppm_tolerance:    Maximum ppm error (default 3.0).
        element_set:      Key into MATRIX_ELEMENT_SETS.
        max_candidates:   Maximum candidates to keep (sorted by ppm error).
        ion_mode:         'negative' or 'positive'.
        inject_formula:   Optional known formula to inject into candidates.
    
    Returns:
        List of candidate dictionaries, each with:
          formula, mass_exact, ppm_error, dbe, element_counts, ...
    """
    # Convert observed m/z to neutral mass
    if ion_mode == 'negative':
        neutral_mass = target_mass + 1.00728
    elif ion_mode == 'positive':
        neutral_mass = target_mass - 1.00728
    else:
        neutral_mass = target_mass
    
    # Absolute tolerance in Da
    tol_da = neutral_mass * ppm_tolerance / 1e6
    mass_lo = neutral_mass - tol_da
    mass_hi = neutral_mass + tol_da
    
    # Get element set
    elements = MATRIX_ELEMENT_SETS.get(element_set, MATRIX_ELEMENT_SETS['DOM'])
    
    # Choose enumeration strategy based on element set
    if element_set in ('PFAS_AFFF',):
        # Multi-pass for complex PFAS element sets
        bail_out = max(max_candidates * 50, 15000)
        candidates = _enumerate_pfas_multipass(mass_lo, mass_hi, neutral_mass, bail_out)
    else:
        # Standard single-pass enumeration
        candidates = _enumerate_nested_loops(
            elements=elements,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
            neutral_mass=neutral_mass,
            bail_out=max(max_candidates * 20, 5000),
        )
    
    # Apply Seven Golden Rules
    valid_candidates = []
    for cand in candidates:
        formula_str = cand['formula']
        if check_golden_rules(formula_str):
            cand['golden_rules_pass'] = True
            valid_candidates.append(cand)
    
    # Inject known formula if not already found (order-independent compare)
    if inject_formula is not None:
        found = any(_formulas_match(c['formula'], inject_formula) for c in valid_candidates)
        if not found:
            injected = _make_candidate_from_formula(inject_formula, neutral_mass)
            if injected is not None:
                valid_candidates.append(injected)
                logger.debug(f"  Injected formula {inject_formula} "
                             f"(ppm={injected['ppm_error']:.2f})")
    
    # Sort by absolute ppm error and cap
    valid_candidates.sort(key=lambda c: abs(c['ppm_error']))
    if len(valid_candidates) > max_candidates:
        valid_candidates = valid_candidates[:max_candidates]
    
    return valid_candidates


def _enumerate_nested_loops(
    elements: List[str],
    mass_lo: float,
    mass_hi: float,
    neutral_mass: float,
    bail_out: int,
) -> List[Dict]:
    """
    Fast dedicated nested-loop enumeration with analytical H-solving.
    
    Separates H (lightest atom) from heavy atoms. For each combination of
    heavy atoms, the valid H count is solved analytically from the remaining
    mass in O(1). Tight per-loop upper bounds avoid visiting infeasible
    combinations. Incremental mass tracking avoids recomputing sums.
    """
    import math
    
    FORMULA_ORDER = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br']
    
    # Separate H from heavy elements
    heavy_elems = [e for e in elements if e != 'H']
    has_h = 'H' in elements
    H_MASS = ATOMIC_MASSES.get('H', 1.0078250321)
    h_min = DEFAULT_BOUNDS['H'][0] if has_h else 0
    h_max_rules = DEFAULT_BOUNDS['H'][1] if has_h else 0
    h_min_mass = h_min * H_MASS
    h_max_mass = h_max_rules * H_MASS
    
    # Effective mass range that heavy atoms must occupy
    heavy_lo = mass_lo - h_max_mass  # min heavy mass (with max H)
    heavy_hi = mass_hi - h_min_mass  # max heavy mass (with min H)
    
    # Get element info
    heavy_masses = [ATOMIC_MASSES[e] for e in heavy_elems]
    heavy_min_counts = [DEFAULT_BOUNDS[e][0] for e in heavy_elems]
    heavy_max_counts = [min(int(mass_hi / ATOMIC_MASSES[e]) + 1, DEFAULT_BOUNDS[e][1])
                        for e in heavy_elems]
    
    candidates = []
    n = len(heavy_elems)
    
    if n == 0:
        return candidates
    
    # Build element index map for quick lookup
    elem_idx = {e: i for i, e in enumerate(heavy_elems)}
    c_idx = elem_idx.get('C', -1)
    
    # Use dedicated nested loops for common element sets (DOM: C,O,N,S)
    # This avoids the overhead of generic odometer and gives ~10x speedup
    
    if set(heavy_elems) == {'C', 'O', 'N', 'S'}:
        _enumerate_dom_fast(
            heavy_masses, heavy_min_counts, heavy_max_counts,
            heavy_elems, elem_idx,
            has_h, H_MASS, h_min, h_max_rules,
            mass_lo, mass_hi, neutral_mass,
            candidates, bail_out,
        )
    elif set(heavy_elems) == {'C', 'O'}:
        _enumerate_cho_fast(
            heavy_masses, heavy_min_counts, heavy_max_counts,
            heavy_elems, elem_idx,
            has_h, H_MASS, h_min, h_max_rules,
            mass_lo, mass_hi, neutral_mass,
            candidates, bail_out,
        )
    elif set(heavy_elems) == {'C', 'O', 'F'}:
        _enumerate_chof_fast(
            heavy_masses, heavy_min_counts, heavy_max_counts,
            heavy_elems, elem_idx,
            has_h, H_MASS, h_min, h_max_rules,
            mass_lo, mass_hi, neutral_mass,
            candidates, bail_out,
        )
    else:
        # Generic fallback using odometer
        _enumerate_generic(
            heavy_elems, heavy_masses, heavy_min_counts, heavy_max_counts,
            has_h, H_MASS, h_min, h_max_rules,
            mass_lo, mass_hi, neutral_mass,
            candidates, bail_out,
        )
    
    return candidates


def _make_candidate(count_map, total_mass, neutral_mass):
    """Helper to build a candidate dict from counts and mass."""
    FORMULA_ORDER = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
    formula_parts = []
    for e in FORMULA_ORDER:
        c = count_map.get(e, 0)
        if c > 0:
            formula_parts.append(f"{e}{c}" if c > 1 else e)
    formula_str = ''.join(formula_parts)
    ppm_err = (total_mass - neutral_mass) / neutral_mass * 1e6
    dbe = compute_dbe(formula_str)
    return {
        'formula': formula_str,
        'counts': count_map,
        'mass_exact': total_mass,
        'ppm_error': ppm_err,
        'dbe': dbe,
        'elemental_class': get_elemental_class(formula_str),
        'dbe_bin': get_dbe_bin(dbe),
    }


def _solve_h(heavy_mass, H_MASS, h_min, h_max_rules, mass_lo, mass_hi):
    """Analytically solve for valid H count range given heavy atom mass."""
    import math
    remaining_lo = mass_lo - heavy_mass
    remaining_hi = mass_hi - heavy_mass
    if remaining_hi < 0:
        return -1, -2  # no valid H (heavy too big)
    if remaining_lo < 0:
        remaining_lo = 0
    h_lo = max(h_min, math.ceil(remaining_lo / H_MASS))
    h_hi = min(h_max_rules, int(remaining_hi / H_MASS))  # floor
    return h_lo, h_hi


def _make_candidate_from_formula(formula_str, neutral_mass):
    """Build a candidate dict from a known formula string (for injection)."""
    counts = parse_formula(formula_str)
    if not counts or counts.get('C', 0) == 0:
        return None
    total_mass = formula_to_mass(formula_str)
    if total_mass is None or total_mass <= 0:
        return None
    ppm_err = (total_mass - neutral_mass) / neutral_mass * 1e6
    dbe = compute_dbe(formula_str)
    return {
        'formula': formula_str,
        'counts': counts,
        'mass_exact': total_mass,
        'ppm_error': ppm_err,
        'dbe': dbe,
        'elemental_class': get_elemental_class(formula_str),
        'dbe_bin': get_dbe_bin(dbe),
        'golden_rules_pass': True,  # injected formulas bypass golden rules
    }


def _enumerate_chof_fast(
    heavy_masses, heavy_min_counts, heavy_max_counts,
    heavy_elems, elem_idx,
    has_h, H_MASS, h_min, h_max_rules,
    mass_lo, mass_hi, neutral_mass,
    candidates, bail_out,
):
    """
    Fast dedicated enumeration for CHOF element set: C, H, O, F.
    Uses 3 nested for-loops (C, O, F) with analytical H-solving.
    Critical for PFAS compounds which have high F counts (F=5-25).
    """
    import math

    C_MASS, O_MASS, F_MASS = [ATOMIC_MASSES[e] for e in ['C', 'O', 'F']]
    c_min, o_min, f_min = [DEFAULT_BOUNDS[e][0] for e in ['C', 'O', 'F']]
    c_max = heavy_max_counts[elem_idx['C']]
    o_max = heavy_max_counts[elem_idx['O']]
    f_max = heavy_max_counts[elem_idx['F']]

    h_min_mass = h_min * H_MASS
    h_max_mass = h_max_rules * H_MASS

    for c in range(c_min, c_max + 1):
        c_mass = c * C_MASS
        if c_mass + h_min_mass > mass_hi:
            break

        o_upper = min(o_max, int((mass_hi - c_mass) / O_MASS))

        for o in range(o_min, o_upper + 1):
            co_mass = c_mass + o * O_MASS
            if co_mass + h_min_mass > mass_hi:
                break

            # Tight F bound
            f_upper = min(f_max, int((mass_hi - co_mass) / F_MASS))
            if has_h:
                f_lower_mass = mass_lo - co_mass - h_max_mass
                f_lower = max(f_min, math.ceil(f_lower_mass / F_MASS)) if f_lower_mass > 0 else f_min
            else:
                f_lower = f_min

            for f in range(f_lower, f_upper + 1):
                cof_mass = co_mass + f * F_MASS

                if cof_mass + h_min_mass > mass_hi:
                    break
                if cof_mass + h_max_mass < mass_lo:
                    continue

                # Solve for H analytically
                if has_h:
                    h_lo, h_hi = _solve_h(cof_mass, H_MASS, h_min, h_max_rules, mass_lo, mass_hi)
                    if h_lo > h_hi:
                        continue

                    for h in range(h_lo, h_hi + 1):
                        total = cof_mass + h * H_MASS
                        count_map = {'C': c, 'H': h, 'O': o, 'F': f}
                        candidates.append(_make_candidate(count_map, total, neutral_mass))
                        if len(candidates) >= bail_out:
                            return
                else:
                    if mass_lo <= cof_mass <= mass_hi:
                        count_map = {'C': c, 'O': o, 'F': f}
                        candidates.append(_make_candidate(count_map, cof_mass, neutral_mass))
                        if len(candidates) >= bail_out:
                            return


def _enumerate_dom_fast(
    heavy_masses, heavy_min_counts, heavy_max_counts,
    heavy_elems, elem_idx,
    has_h, H_MASS, h_min, h_max_rules,
    mass_lo, mass_hi, neutral_mass,
    candidates, bail_out,
):
    """
    Fast dedicated enumeration for DOM element set: C, H, O, N, S.
    Uses 4 nested for-loops with tight per-level bounds.
    """
    import math
    
    C_MASS, O_MASS, N_MASS, S_MASS = [ATOMIC_MASSES[e] for e in ['C', 'O', 'N', 'S']]
    c_min, o_min, n_min, s_min = [DEFAULT_BOUNDS[e][0] for e in ['C', 'O', 'N', 'S']]
    c_max = heavy_max_counts[elem_idx['C']]
    o_max = heavy_max_counts[elem_idx['O']]
    n_max = heavy_max_counts[elem_idx['N']]
    s_max = heavy_max_counts[elem_idx['S']]
    
    h_min_mass = h_min * H_MASS
    h_max_mass = h_max_rules * H_MASS
    
    for c in range(c_min, c_max + 1):
        c_mass = c * C_MASS
        if c_mass + h_min_mass > mass_hi:
            break  # C is too heavy; larger C impossible
        
        # Tight O bound for this C
        o_upper = min(o_max, int((mass_hi - c_mass) / O_MASS))
        
        for o in range(o_min, o_upper + 1):
            co_mass = c_mass + o * O_MASS
            if co_mass + h_min_mass > mass_hi:
                break
            
            # Tight N bound
            n_upper = min(n_max, int((mass_hi - co_mass) / N_MASS))
            
            for n in range(n_min, n_upper + 1):
                con_mass = co_mass + n * N_MASS
                if con_mass + h_min_mass > mass_hi:
                    break
                
                # Tight S bound
                s_upper = min(s_max, int((mass_hi - con_mass) / S_MASS))
                
                # Also compute minimum S to be heavy enough
                if has_h:
                    s_lower_mass = mass_lo - con_mass - h_max_mass
                    s_lower = max(s_min, math.ceil(s_lower_mass / S_MASS)) if s_lower_mass > 0 else s_min
                else:
                    s_lower = s_min
                
                for s in range(s_lower, s_upper + 1):
                    cons_mass = con_mass + s * S_MASS
                    
                    if cons_mass + h_min_mass > mass_hi:
                        break
                    if cons_mass + h_max_mass < mass_lo:
                        continue
                    
                    # Solve for H analytically
                    if has_h:
                        h_lo, h_hi = _solve_h(cons_mass, H_MASS, h_min, h_max_rules, mass_lo, mass_hi)
                        if h_lo > h_hi:
                            continue
                        
                        for h in range(h_lo, h_hi + 1):
                            total = cons_mass + h * H_MASS
                            count_map = {'C': c, 'H': h, 'O': o, 'N': n, 'S': s}
                            candidates.append(_make_candidate(count_map, total, neutral_mass))
                            if len(candidates) >= bail_out:
                                return
                    else:
                        if mass_lo <= cons_mass <= mass_hi:
                            count_map = {'C': c, 'O': o, 'N': n, 'S': s}
                            candidates.append(_make_candidate(count_map, cons_mass, neutral_mass))
                            if len(candidates) >= bail_out:
                                return


def _enumerate_cho_fast(
    heavy_masses, heavy_min_counts, heavy_max_counts,
    heavy_elems, elem_idx,
    has_h, H_MASS, h_min, h_max_rules,
    mass_lo, mass_hi, neutral_mass,
    candidates, bail_out,
):
    """Fast dedicated enumeration for CHO element set."""
    import math
    
    C_MASS = ATOMIC_MASSES['C']
    O_MASS = ATOMIC_MASSES['O']
    c_min = DEFAULT_BOUNDS['C'][0]
    o_min = DEFAULT_BOUNDS['O'][0]
    c_max = heavy_max_counts[elem_idx['C']]
    o_max = heavy_max_counts[elem_idx['O']]
    
    for c in range(c_min, c_max + 1):
        c_mass = c * C_MASS
        if c_mass > mass_hi:
            break
        
        o_upper = min(o_max, int((mass_hi - c_mass) / O_MASS))
        
        for o in range(o_min, o_upper + 1):
            co_mass = c_mass + o * O_MASS
            if co_mass > mass_hi:
                break
            
            if has_h:
                h_lo, h_hi = _solve_h(co_mass, H_MASS, h_min, h_max_rules, mass_lo, mass_hi)
                if h_lo > h_hi:
                    continue
                for h in range(h_lo, h_hi + 1):
                    total = co_mass + h * H_MASS
                    count_map = {'C': c, 'H': h, 'O': o}
                    candidates.append(_make_candidate(count_map, total, neutral_mass))
                    if len(candidates) >= bail_out:
                        return


def _enumerate_generic(
    heavy_elems, heavy_masses, heavy_min_counts, heavy_max_counts,
    has_h, H_MASS, h_min, h_max_rules,
    mass_lo, mass_hi, neutral_mass,
    candidates, bail_out,
):
    """Generic odometer-based enumeration for arbitrary element sets."""
    import math
    
    n_heavy = len(heavy_elems)
    FORMULA_ORDER = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br']
    
    h_min_mass = h_min * H_MASS if has_h else 0.0
    h_max_mass = h_max_rules * H_MASS if has_h else 0.0
    
    # Pre-compute suffix bounds
    suffix_min_mass = [0.0] * (n_heavy + 1)
    suffix_max_mass = [0.0] * (n_heavy + 1)
    for d in range(n_heavy - 1, -1, -1):
        suffix_min_mass[d] = suffix_min_mass[d + 1] + heavy_masses[d] * heavy_min_counts[d]
        suffix_max_mass[d] = suffix_max_mass[d + 1] + heavy_masses[d] * heavy_max_counts[d]
    
    # C index for carbon check
    c_idx = None
    for i, e in enumerate(heavy_elems):
        if e == 'C':
            c_idx = i
            break
    
    counts = list(heavy_min_counts)
    
    while True:
        if len(candidates) >= bail_out:
            break
        
        heavy_mass = sum(counts[i] * heavy_masses[i] for i in range(n_heavy))
        
        if heavy_mass + h_min_mass <= mass_hi and heavy_mass + h_max_mass >= mass_lo:
            has_carbon = c_idx is not None and counts[c_idx] > 0
            if has_carbon:
                if has_h:
                    h_lo, h_hi = _solve_h(heavy_mass, H_MASS, h_min, h_max_rules, mass_lo, mass_hi)
                    for h in range(h_lo, h_hi + 1):
                        total = heavy_mass + h * H_MASS
                        count_map = {heavy_elems[i]: counts[i] for i in range(n_heavy)}
                        count_map['H'] = h
                        candidates.append(_make_candidate(count_map, total, neutral_mass))
                        if len(candidates) >= bail_out:
                            return
                elif mass_lo <= heavy_mass <= mass_hi:
                    count_map = {heavy_elems[i]: counts[i] for i in range(n_heavy)}
                    candidates.append(_make_candidate(count_map, heavy_mass, neutral_mass))
        
        # Increment odometer
        carry = True
        for d in range(n_heavy - 1, -1, -1):
            if not carry:
                break
            counts[d] += 1
            if counts[d] > heavy_max_counts[d]:
                counts[d] = heavy_min_counts[d]
                carry = True
                continue
            carry = False
            partial = sum(counts[i] * heavy_masses[i] for i in range(d + 1))
            min_rest = suffix_min_mass[d + 1]
            max_rest = suffix_max_mass[d + 1]
            if partial + min_rest + h_min_mass > mass_hi:
                counts[d] = heavy_min_counts[d]
                carry = True
                continue
            elif partial + max_rest + h_max_mass < mass_lo:
                upper = sum(counts[i] * heavy_masses[i] for i in range(d))
                needed = mass_lo - upper - max_rest - h_max_mass
                if heavy_masses[d] > 0:
                    skip_to = max(counts[d], math.ceil(needed / heavy_masses[d]))
                else:
                    skip_to = heavy_max_counts[d] + 1
                if skip_to <= heavy_max_counts[d]:
                    counts[d] = skip_to
                else:
                    counts[d] = heavy_min_counts[d]
                    carry = True
                    continue
        
        if carry:
            break


def _enumerate_pfas_multipass(mass_lo, mass_hi, neutral_mass, bail_out):
    """
    Multi-pass enumeration for PFAS_AFFF element set.

    Instead of running one generic 8-element odometer (which exhausts
    bail_out on low-F candidates), we enumerate with progressively larger
    element sub-sets, each using the appropriate fast path:

      Pass 1: C,H,O,F         — core PFAS (PFOA, PFHxA, GenX, FTOHs)
      Pass 2: C,H,O,F,S       — sulfonated PFAS (PFOS, PFHxS, FTS)
      Pass 3: C,H,O,F,N       — N-substituted PFAS (FOSA)
      Pass 4: C,H,O,N,S       — DOM contaminants
      Pass 5: C,H,O,F,N,S     — complex PFAS (FOSA with S)
      Pass 6: C,H,O,F,S,P     — phosphonate PFAS

    Results are merged and deduplicated by formula string.
    """
    all_candidates = []
    seen_formulas = set()

    # Sub-element sets ordered by PFAS relevance
    sub_sets = [
        ['C', 'H', 'O', 'F'],             # Core PFAS
        ['C', 'H', 'O', 'F', 'S'],        # Sulfonated PFAS
        ['C', 'H', 'O', 'F', 'N'],        # N-containing PFAS
        ['C', 'H', 'O', 'N', 'S'],        # DOM contaminants
        ['C', 'H', 'O', 'F', 'N', 'S'],   # Complex PFAS
        ['C', 'H', 'O', 'F', 'S', 'P'],   # P-containing PFAS
    ]

    per_pass_bail = max(bail_out // len(sub_sets), 2000)

    for sub_elems in sub_sets:
        partial_candidates = _enumerate_nested_loops(
            elements=sub_elems,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
            neutral_mass=neutral_mass,
            bail_out=per_pass_bail,
        )
        n_new = 0
        for c in partial_candidates:
            if c['formula'] not in seen_formulas:
                seen_formulas.add(c['formula'])
                all_candidates.append(c)
                n_new += 1

    logger.debug(f"  PFAS multi-pass: {len(all_candidates)} unique candidates "
                 f"from {len(sub_sets)} passes")
    return all_candidates


def build_fcl_graph(
    candidates: List[Dict],
    h_mix: Optional[torch.Tensor] = None,
    p_iso: Optional[torch.Tensor] = None,
) -> Data:
    """
    Build Formula Candidate Lattice Graph (G_FCL) from enumerated candidates.
    
    Nodes = candidate formulas
    Edges = connect candidates differing by ±1 unit element
    
    Node features (17-dim):
      [0]  DBE
      [1]  O/C ratio
      [2]  H/C ratio
      [3]  N/C ratio
      [4]  S/C ratio
      [5]  F/C ratio
      [6]  ppm_error
      [7]  mass_exact (normalised /1000)
      [8]  valence_ok (golden rules pass)
      [9]  electron_parity
      [10] A_norm          (Bethe-Weizsäcker: total nucleons / 1000)
      [11] Z_norm          (BW: total protons / 500)
      [12] mass_defect_norm (BW: mass defect / 10)
      [13] BE_per_A_norm   (BW: binding energy per nucleon / 10)
      [14] N×S interaction  (n_N * n_S / 10)  — targets CHONS failure mode
      [15] P indicator      (1 if P present)  — targets CHOPN/CHONSP failure
      [16] N×P interaction  (n_N * n_P / 5)   — targets CHONSP failure mode
    
    Optionally concatenate h_mix (256-dim mixture context) and p_iso (9-dim).
    
    Args:
        candidates: List of candidate dictionaries from enumerate_candidates.
        h_mix:      Optional mixture context embedding (256-dim).
        p_iso:      Optional isotope probability vector (9-dim).
    
    Returns:
        PyG Data object with lattice structure.
    """
    N = len(candidates)
    if N == 0:
        data = Data(
            x=torch.zeros((0, 17), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
            num_nodes=0,
        )
        data.formula_labels = []
        data.elemental_classes = []
        data.dbe_bins = []
        data.formula_vectors = torch.zeros((0, len(ELEMENTS)), dtype=torch.float32)
        return data
    
    # ── Build node features ──
    features = np.zeros((N, 17), dtype=np.float32)  # 10 original + 4 BW + 3 interaction
    formula_vectors = np.zeros((N, len(ELEMENTS)), dtype=np.float32)
    
    for i, cand in enumerate(candidates):
        counts = cand['counts']
        c = counts.get('C', 0)
        h = counts.get('H', 0)
        o = counts.get('O', 0)
        n_count = counts.get('N', 0)
        s = counts.get('S', 0)
        f = counts.get('F', 0)
        p = counts.get('P', 0)
        
        features[i, 0] = cand['dbe']
        features[i, 1] = o / max(c, 1)          # O/C
        features[i, 2] = h / max(c, 1)          # H/C
        features[i, 3] = n_count / max(c, 1)    # N/C
        features[i, 4] = s / max(c, 1)          # S/C
        features[i, 5] = f / max(c, 1)          # F/C
        features[i, 6] = cand['ppm_error']
        features[i, 7] = cand['mass_exact'] / 1000.0  # Normalized mass
        features[i, 8] = 1.0 if cand.get('golden_rules_pass', True) else 0.0  # valence_ok
        features[i, 9] = 1.0 if cand['dbe'] == int(cand['dbe']) else 0.0      # electron parity
        
        # BW features (4-dim): [A_norm, Z_norm, mass_defect_norm, BE/A_norm]
        features[i, 10:14] = compute_bw_features(counts)
        
        # Interaction features — directly encode failure-mode signals
        features[i, 14] = (n_count * s) / 10.0          # N×S product (CHONS target)
        features[i, 15] = 1.0 if p > 0 else 0.0         # P presence indicator (CHOPN target)
        features[i, 16] = (n_count * p) / 5.0            # N×P product (CHONSP target)
        
        # Store element vector for lattice edge computation
        for k, elem in enumerate(ELEMENTS):
            formula_vectors[i, k] = counts.get(elem, 0)
    
    # ── Build lattice edges ──
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_delta_type: List[int] = []
    
    # For efficiency, build lookup from formula vector signature
    for i in range(N):
        for j in range(i + 1, N):
            diff = formula_vectors[j] - formula_vectors[i]
            abs_diff = np.abs(diff)
            total_diff = abs_diff.sum()
            
            # Connect if they differ by exactly one unit element operation
            # (e.g., +C, -C, +2H, -2H, +O, -O, +N, -N, +S, -S)
            match = False
            delta_type = -1
            
            if total_diff == 1:
                # Differ by ±1 of a single element
                idx = np.argmax(abs_diff)
                if abs_diff[idx] == 1:
                    match = True
                    delta_type = idx * 2 + (0 if diff[idx] > 0 else 1)
            
            elif total_diff == 2 and abs_diff[1] == 2:
                # Differ by ±2H (maintain electron parity)
                match = True
                delta_type = 2 * 2 + (0 if diff[1] > 0 else 1)  # H+2 or H-2
            
            if match:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_delta_type.extend([delta_type, delta_type])
    
    # ── Construct PyG Data ──
    x = torch.tensor(features, dtype=torch.float32)
    
    # Optionally concatenate conditioning vectors
    if h_mix is not None:
        # Broadcast h_mix to all candidates
        h_mix_expanded = h_mix.unsqueeze(0).expand(N, -1)
        x = torch.cat([x, h_mix_expanded], dim=-1)
    
    if p_iso is not None:
        p_iso_expanded = p_iso.unsqueeze(0).expand(N, -1)
        x = torch.cat([x, p_iso_expanded], dim=-1)
    
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_delta_type, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    
    # Store formula labels for training
    formula_labels = [cand['formula'] for cand in candidates]
    elemental_classes = [cand['elemental_class'] for cand in candidates]
    dbe_bins = [cand['dbe_bin'] for cand in candidates]
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=N,
    )
    
    # Store metadata as Python attributes (not tensors)
    data.formula_labels = formula_labels
    data.elemental_classes = elemental_classes
    data.dbe_bins = dbe_bins
    data.formula_vectors = torch.tensor(formula_vectors, dtype=torch.float32)
    
    n_edges = len(edge_src) // 2
    logger.debug(f"  FCL: {N} candidates, {n_edges} lattice edges")
    
    return data


def build_fcl_for_sample(
    mz_array: np.ndarray,
    ppm_tolerance: float = 3.0,
    element_set: str = 'DOM',
    max_candidates: int = 500,
    ion_mode: str = 'negative',
    true_formulas: Optional[List[str]] = None,
    inject_formulas: Optional[List[str]] = None,
) -> Tuple[List[Data], Dict]:
    """
    Build FCL graphs for all features in a sample.
    
    Args:
        mz_array:         m/z values of all features.
        ppm_tolerance:    Maximum ppm error for candidate enumeration.
        element_set:      Element set for enumeration.
        max_candidates:   Maximum candidates per feature.
        ion_mode:         'negative' or 'positive'.
        true_formulas:    Optional ground truth formulas for validation.
        inject_formulas:  Optional formulas to inject into candidate sets,
                          ensuring ground truth is always present for training.
    
    Returns:
        Tuple of:
          - List of FCL Data objects (one per feature)
          - Statistics dictionary
    """
    import time
    N = len(mz_array)
    fcl_graphs: List[Data] = []
    
    total_candidates = 0
    correct_in_set_count = 0
    
    t0 = time.time()
    log_interval = max(N // 5, 100)  # Log progress ~5 times
    
    for i in range(N):
        # Determine injection formula for this feature
        inject_f = None
        if inject_formulas is not None and i < len(inject_formulas):
            inject_f = inject_formulas[i]
        
        candidates = enumerate_candidates(
            target_mass=mz_array[i],
            ppm_tolerance=ppm_tolerance,
            element_set=element_set,
            max_candidates=max_candidates,
            ion_mode=ion_mode,
            inject_formula=inject_f,
        )
        
        fcl_data = build_fcl_graph(candidates)
        fcl_data.feature_idx = i
        fcl_data.observed_mz = mz_array[i]
        fcl_graphs.append(fcl_data)
        
        total_candidates += len(candidates)
        
        # Check if true formula is in candidate set (order-independent)
        if true_formulas is not None and i < len(true_formulas):
            true_f = true_formulas[i]
            if any(_formulas_match(c['formula'], true_f) for c in candidates):
                correct_in_set_count += 1
        
        # Progress logging
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"    FCL progress: {i+1}/{N} features "
                f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining, "
                f"avg {total_candidates/(i+1):.1f} cands/feat)"
            )
    
    mean_candidates = total_candidates / max(N, 1)
    elapsed_total = time.time() - t0
    
    stats = {
        'total_features': N,
        'total_candidates': total_candidates,
        'mean_candidates_per_feature': mean_candidates,
    }
    
    if true_formulas is not None:
        stats['correct_in_candidate_set'] = correct_in_set_count
        stats['recall'] = correct_in_set_count / max(N, 1)
    
    logger.info(
        f"  FCL: {N} features, {total_candidates} total candidates, "
        f"mean {mean_candidates:.1f} per feature ({elapsed_total:.1f}s)"
    )
    
    return fcl_graphs, stats
