"""
P2-T2: Module 2 — Isotope GAT Encoder + Roepstorff Physics Layer
=================================================================
Graph Attention Network on per-feature isotopologue sub-graphs.

Each monoisotopic feature has a small sub-graph connecting it to
its M+1, M+2 isotope peaks.  The GAT encodes these into a
heteroatom probability vector p_iso ∈ R^9 (C,H,O,N,S,P,F,Cl,Br),
plus a differentiable Roepstorff physics layer that predicts
theoretical isotope ratios from atom counts, enabling a physics-
residual loss (MSE observed vs. theoretical ratios).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, List, Optional, Tuple
import math


# ─────────────────────────────────────────────────────────────
# Differentiable Roepstorff Physics Layer
# ─────────────────────────────────────────────────────────────

class DifferentiableRoepstorff(nn.Module):
    """
    Differentiable isotope intensity prediction based on the
    Roepstorff/binomial expansion model.

    Given atom counts n_elem and natural abundances p_elem,
    computes theoretical I(M+k)/I(M) via:

        ratio_k = Σ_elem C(n_elem, k) · p^k · (1-p)^(n-k)

    Made differentiable by operating on continuous atom-count
    estimates (soft counts from the network output).
    """

    def __init__(self):
        super().__init__()
        # Natural isotope abundances (heavy / light)
        self.register_buffer('abundances', torch.tensor([
            0.01103,   # 13C / 12C
            0.000115,  # 2H / 1H  (negligible but included)
            0.00038,   # 17O / 16O (for M+1 only)
            0.00368,   # 15N / 14N
            0.00750,   # 33S / 32S  (M+1)
            0.0,       # P  (no significant isotopes)
            0.0,       # F  (monoisotopic)
            0.24229,   # 37Cl / 35Cl
            0.49314,   # 81Br / 79Br
        ], dtype=torch.float32))

        # M+2 abundances for elements with significant +2 shift
        self.register_buffer('abundances_m2', torch.tensor([
            0.01103,   # 13C (two 13C)
            0.0,       # H
            0.00205,   # 18O / 16O
            0.0,       # N
            0.04250,   # 34S / 32S
            0.0,       # P
            0.0,       # F
            0.24229,   # 37Cl (M+2 from two Cl)
            0.49314,   # 81Br (M+2 from two Br)
        ], dtype=torch.float32))

    def forward(
        self,
        atom_counts: torch.Tensor,
        k: int = 1,
    ) -> torch.Tensor:
        """
        Compute theoretical I(M+k)/I(M) ratio.

        Args:
            atom_counts: [batch, 9] — continuous atom count estimates
                         for [C, H, O, N, S, P, F, Cl, Br].
            k: isotopologue order (1 or 2).

        Returns:
            ratios: [batch] — predicted I(M+k)/I(M) values.
        """
        # Clamp counts to positive values
        n = atom_counts.clamp(min=0.0)

        if k == 1:
            p = self.abundances.unsqueeze(0)  # [1, 9]
            # Approximate binomial: C(n,1) * p * (1-p)^(n-1) = n * p * (1-p)^(n-1)
            contrib = n * p * (1.0 - p).pow(n - 1.0 + 1e-8)
            ratio = contrib.sum(dim=-1)

        elif k == 2:
            p2 = self.abundances_m2.unsqueeze(0)  # [1, 9]
            # Two contributions to M+2:
            # 1) Same element, two heavy atoms: C(n,2) * p^2 * (1-p)^(n-2)
            contrib_same = 0.5 * n * (n - 1).clamp(min=0) * p2 * p2 * \
                           (1.0 - p2).pow((n - 2).clamp(min=0))
            # 2) Direct M+2 isotopes (34S, 18O, 37Cl, 81Br):
            #    C(n,1) * p_m2 for elements where M+2 shift is a single heavy isotope
            direct_m2 = n * p2 * (1.0 - p2).pow(n - 1.0 + 1e-8)
            # For C: the M+2 contribution is from two 13C atoms
            # For S,Cl,Br: from one 34S/37Cl/81Br (direct_m2)
            # Simplified: sum both contributions
            ratio = (contrib_same + direct_m2).sum(dim=-1)
        else:
            ratio = torch.zeros(atom_counts.size(0), device=atom_counts.device)

        return ratio


# ─────────────────────────────────────────────────────────────
# Isotope GAT Encoder
# ─────────────────────────────────────────────────────────────

class IsotopeGATEncoder(nn.Module):
    """
    Module 2: GAT encoder on isotopologue sub-graphs.

    Architecture:
        GAT(3 layers, 8 heads, hidden=128)
        → mean pool over sub-graph
        → MLP(128 → 64 → 9)  →  sigmoid  →  p_iso

    Input : Per-feature isotope sub-graphs (Data objects with
            4-dim node features: [m/z, log10(int+1), obs_ratio, is_mono]).
    Output: p_iso ∈ R^{N_features × 9} — heteroatom probabilities.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 9,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            dim_in = hidden_dim
            # GATConv: heads output is concatenated, so output per head = hidden//heads
            head_dim = hidden_dim // num_heads
            self.convs.append(
                GATConv(dim_in, head_dim, heads=num_heads,
                        dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # MLP head → heteroatom probabilities
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Roepstorff physics layer
        self.roepstorff = DifferentiableRoepstorff()

    def forward(
        self,
        iso_batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            iso_batch: Batched isotope sub-graphs (from Batch.from_data_list).

        Returns:
            p_iso:        [B, 9] — heteroatom probabilities.
            theo_m1:      [B]    — predicted I(M+1)/I(M) from Roepstorff.
            theo_m2:      [B]    — predicted I(M+2)/I(M) from Roepstorff.
        """
        x = iso_batch.x
        edge_index = iso_batch.edge_index
        batch = iso_batch.batch

        # Input projection
        h = self.input_proj(x)

        # GAT layers with residual + LayerNorm
        for i in range(len(self.convs)):
            h_res = h
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + h_res  # residual

        # Global mean pooling per sub-graph → [B, hidden_dim]
        h_pool = global_mean_pool(h, batch)

        # Heteroatom probability prediction
        p_iso = self.head(h_pool)  # [B, 9], range [0, 1]

        # Roepstorff physics: scale p_iso by max atom counts to get estimated counts
        # p_iso * max_atoms gives approximate atom counts
        # Use reasonable max bounds: [C≤50, H≤100, O≤30, N≤10, S≤5, P≤5, F≤30, Cl≤10, Br≤5]
        max_atoms = torch.tensor(
            [50.0, 100.0, 30.0, 10.0, 5.0, 5.0, 30.0, 10.0, 5.0],
            device=p_iso.device,
        )
        atom_counts_est = p_iso * max_atoms.unsqueeze(0)

        theo_m1 = self.roepstorff(atom_counts_est, k=1)
        theo_m2 = self.roepstorff(atom_counts_est, k=2)

        return p_iso, theo_m1, theo_m2

    def forward_single(self, iso_data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single isotope sub-graph (unbatched)."""
        batch = Batch.from_data_list([iso_data])
        batch = batch.to(next(self.parameters()).device)
        return self.forward(batch)
