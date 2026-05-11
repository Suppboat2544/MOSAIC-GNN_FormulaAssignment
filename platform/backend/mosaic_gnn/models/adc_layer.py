"""
P2-T4: Module 4 — Adduct Consistency Differentiable Layer
==========================================================
Soft-corrects formula scoring by enforcing that adduct-connected
features (e.g., [M+H]⁺ and [M+Na]⁺ of the same compound) must
agree on the neutral molecular formula.

Implements a differentiable penalty:
    s_ij_corrected = s_ij − λ · L_ADC

where L_ADC measures neutral-mass disagreement between
adduct-linked features.  λ is a learnable parameter
(initialized at 0.1, annealed to 0.5 during training).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple


# Standard adduct offsets used to recover neutral mass from m/z
ADDUCT_MASS_OFFSETS = {
    '[M-H]-':   -1.00728,
    '[M+Cl]-':  34.96941,
    '[M+H]+':    1.00728,
    '[M+Na]+':  22.98922,
    '[M+K]+':   38.96316,
    '[M+NH4]+': 18.03437,
}


class AdductConsistencyLayer(nn.Module):
    """
    Module 4: Differentiable adduct consistency correction.

    For each adduct-connected pair (i, k) in G_ADC:
      1. Retrieve top-1 candidate formula scores s_i, s_k
      2. Compute neutral mass: M_i = mz_i − δ_adduct_i
      3. Consistency penalty: L_pair = (M_i − M_k)²
      4. Correct scores: s_corrected = s − λ · L_pair

    Args:
        lambda_init: Initial correction strength (default 0.1).
        learnable_lambda: Whether λ is a trainable parameter.
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        learnable_lambda: bool = True,
    ):
        super().__init__()
        if learnable_lambda:
            self.lam = nn.Parameter(torch.tensor(lambda_init))
        else:
            self.register_buffer('lam', torch.tensor(lambda_init))

    def forward(
        self,
        energies: torch.Tensor,
        candidate_masses: torch.Tensor,
        feature_mz: torch.Tensor,
        adc_edge_index: torch.Tensor,
        adc_edge_type: Optional[torch.Tensor] = None,
        candidate_to_feature: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adduct consistency correction to candidate energies.

        Args:
            energies:            [total_candidates] — raw energy scores from GIN.
            candidate_masses:    [total_candidates] — exact mass of each candidate formula.
            feature_mz:          [N_features] — observed m/z per feature.
            adc_edge_index:      [2, E_adc] — adduct edges between features.
            adc_edge_type:       [E_adc] — adduct type index per edge (optional).
            candidate_to_feature: [total_candidates] — feature index for each candidate.

        Returns:
            corrected_energies: [total_candidates] — consistency-corrected scores.
            L_adc:              scalar — total adduct consistency loss.
        """
        if adc_edge_index.numel() == 0 or adc_edge_index.size(1) == 0:
            return energies, torch.tensor(0.0, device=energies.device)

        N_features = feature_mz.size(0)

        # ── Vectorised top-1 candidate per feature (replaces Python for-loop) ──
        # For each feature, find the candidate with the lowest energy (= best)
        # and retrieve its mass.
        top1_mass = torch.zeros(N_features, device=energies.device)
        if candidate_to_feature is not None:
            # Fill unseen features with +inf so scatter_reduce amax works
            INF = torch.tensor(float('inf'), device=energies.device)
            # scatter_reduce 'amin' on energies, keyed by feature
            min_energy = INF.expand(N_features).clone()
            min_energy.scatter_reduce_(
                0, candidate_to_feature, energies, reduce='amin',
                include_self=True,
            )
            # Boolean mask: which candidate equals the min for its feature?
            is_min = (energies == min_energy[candidate_to_feature])
            # Among ties pick the first one (multiply by descending index)
            # Using scatter to write mass of min-energy candidate:
            # We write ALL candidates matching min then the last write wins;
            # that is fine — ties mean the mass is equally good.
            top1_mass.scatter_(0, candidate_to_feature[is_min],
                               candidate_masses[is_min])

        # Compute neutral mass for each feature's top-1 candidate
        # (Simplified: assume [M-H]⁻ adduct for negative mode → M = mz + 1.00728)
        # In a full implementation, the adduct type per feature would be used here
        neutral_mass = feature_mz + 1.00728  # default for [M-H]⁻

        # Consistency penalty across adduct pairs
        src, dst = adc_edge_index[0], adc_edge_index[1]
        M_src = top1_mass[src]
        M_dst = top1_mass[dst]

        # Penalty: squared neutral mass disagreement
        pair_penalty = (M_src - M_dst) ** 2

        # Total consistency loss
        L_adc = pair_penalty.mean()

        # Correct energies: increase energy (= lower probability) for features
        # in inconsistent adduct pairs
        penalty_per_feature = torch.zeros(N_features, device=energies.device)
        penalty_per_feature.scatter_add_(0, src, pair_penalty)
        # Normalize by degree
        degree = torch.zeros(N_features, device=energies.device)
        degree.scatter_add_(0, src, torch.ones_like(pair_penalty))
        penalty_per_feature = penalty_per_feature / (degree + 1e-8)

        # Broadcast to candidates
        if candidate_to_feature is not None:
            candidate_penalty = penalty_per_feature[candidate_to_feature]
        else:
            candidate_penalty = torch.zeros_like(energies)

        corrected_energies = energies + self.lam.abs() * candidate_penalty

        return corrected_energies, L_adc

    def get_lambda(self) -> float:
        """Return current lambda value."""
        return self.lam.abs().item()
