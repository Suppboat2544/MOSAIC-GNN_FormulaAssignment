"""
P2-T3: Module 3 — Formula Candidate Lattice GIN Scorer
=======================================================
Graph Isomorphism Network (GIN) that scores candidate molecular
formulas for each MS1 feature.

Each feature has a small lattice graph G_FCL whose nodes are
candidate formulas connected by unit-element operations.
The GIN incorporates conditioning from:
  - h_mix (Module 1 SMDG embedding, 256-dim)
  - p_iso (Module 2 isotope probabilities, 9-dim)

Output: Energy scores E(F_ij) per candidate, converted to
        P(F_ij | v_i, S) via EBM softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from typing import Dict, List, Optional, Tuple


class GINLayer(nn.Module):
    """
    Single GIN layer: h^(l+1) = MLP((1+ε) · h^(l) + Σ_{j∈N(i)} h_j^(l))
    with learnable ε.
    """

    def __init__(self, in_dim: int, out_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.eps = nn.Parameter(torch.tensor(eps_init))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Aggregate neighbor features
        row, col = edge_index
        N = x.size(0)
        agg = torch.zeros(N, x.size(1), device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])

        # GIN update
        out = (1.0 + self.eps) * x + agg
        return self.mlp(out)


class FormulaGINScorer(nn.Module):
    """
    Module 3: GIN-based energy scorer for formula candidate lattices.

    Architecture:
        Input:  [formula_features(10) | h_mix(256) | p_iso(9)] = 275-dim
        GIN:    3 layers, hidden_dim=256, learnable eps
        Readout: MLP(256 → 128 → 1) per node → energy score

    Scoring:
        E(F_ij) = readout(h_ij^final)
        P(F_ij) = softmax(-E) = exp(-E_ij) / Σ_k exp(-E_ik)
    """

    def __init__(
        self,
        formula_feat_dim: int = 14,
        h_mix_dim: int = 256,
        p_iso_dim: int = 9,
        h_dag_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 3,
        eps_init: float = 0.0,
        dropout: float = 0.1,
        use_bypass: bool = True,
    ):
        super().__init__()
        self.formula_feat_dim = formula_feat_dim
        self.h_mix_dim = h_mix_dim
        self.p_iso_dim = p_iso_dim
        self.h_dag_dim = h_dag_dim
        input_dim = formula_feat_dim + h_mix_dim + p_iso_dim  # 279 (14+256+9)

        self.use_bypass = use_bypass

        if self.use_bypass:
            # Formula feature projection: lifts formula features (14-dim) to
            # a comparable dimensionality as h_mix (256) so they are not
            # drowned out after concatenation.
            self.formula_proj = nn.Sequential(
                nn.Linear(formula_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
            )
            proj_input_dim = 128 + h_mix_dim + p_iso_dim  # 393
        else:
            proj_input_dim = input_dim  # 279

        # Optional DAG-GNN embedding concatenation (P3-T5)
        if h_dag_dim > 0:
            proj_input_dim += h_dag_dim  # +128 from DAG-GNN

        # Input projection
        self.input_proj = nn.Linear(proj_input_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINLayer(hidden_dim, hidden_dim, eps_init))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Energy readout MLP: 256 → 128 → 1
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Learnable energy scale — amplifies energy differences so that
        # softmax(-E) produces non-uniform probabilities.
        # Initialized > 1 to start with meaningful probability separation.
        if self.use_bypass:
            self.energy_scale = nn.Parameter(torch.tensor(10.0))

            # ── Formula bypass pathway ─────────────────────────────────
            # Direct scoring from projected formula features, bypassing the
            # GIN entirely.  This prevents "energy collapse".
            self.formula_bypass = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            # Feature–formula interaction: project h_mix into formula-embedding
            # space so a dot-product score captures which formula fits which
            # MS1 feature.
            self.mix_proj = nn.Linear(h_mix_dim, 128)

        # Gradient checkpointing — saves ~60% GIN activation memory
        self.use_checkpointing = True

    def _build_input(
        self,
        fcl_batch: Batch,
        h_mix: Optional[torch.Tensor] = None,
        p_iso: Optional[torch.Tensor] = None,
        h_dag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build GIN input tensor from formula features + conditioning.

        Uses formula_proj to lift 14-dim formula features to 128-dim
        before concatenation with h_mix (256-dim), p_iso (9-dim),
        and optionally h_dag (128-dim from DAG-GNN).
        This ensures formula differences between candidates are visible
        and not drowned out by the shared h_mix signal.

        Returns:
            x:          [total_nodes, proj_input_dim] concatenated input
            edge_index: [2, E]
            batch:      [total_nodes]
            formula_h:  [total_nodes, 128] projected formula features
                        (used by bypass pathway in forward)
        """
        x = fcl_batch.x
        edge_index = fcl_batch.edge_index
        batch = fcl_batch.batch
        feat_dim = x.size(1)

        if feat_dim == self.formula_feat_dim and (h_mix is not None or p_iso is not None):
            if self.use_bypass:
                # Project formula features: 14-dim → 128-dim
                formula_h = self.formula_proj(x)  # [N, 128]
            else:
                formula_h = None

            parts = [formula_h if formula_h is not None else x]
            if h_mix is not None:
                parts.append(h_mix[batch])          # [N, 256]
            else:
                parts.append(torch.zeros(x.size(0), self.h_mix_dim, device=x.device))
            if p_iso is not None:
                parts.append(p_iso[batch])           # [N, 9]
            else:
                parts.append(torch.zeros(x.size(0), self.p_iso_dim, device=x.device))
            # Optional DAG-GNN embedding (P3-T5)
            if self.h_dag_dim > 0:
                if h_dag is not None:
                    parts.append(h_dag[batch])       # [N, 128]
                else:
                    parts.append(torch.zeros(x.size(0), self.h_dag_dim, device=x.device))
            x = torch.cat(parts, dim=-1)
            del parts
        else:
            # Fallback: if formula features were already expanded,
            # check if formula_proj needs to be applied
            if self.use_bypass and feat_dim >= self.formula_feat_dim:
                # Split and project formula portion
                formula_part = x[:, :self.formula_feat_dim]
                rest_part = x[:, self.formula_feat_dim:]
                formula_h = self.formula_proj(formula_part)
                x = torch.cat([formula_h, rest_part], dim=-1)
            else:
                formula_h = None
                # Edge case: features smaller than expected
                if not self.use_bypass and feat_dim < self.formula_feat_dim:
                    formula_h = None

        return x, edge_index, batch, formula_h

    def _run_gin(
        self, x: torch.Tensor, edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run input_proj + GIN conv layers.  Returns h [total_nodes, hidden_dim].
        Uses gradient checkpointing when self.use_checkpointing is True
        to trade ~30% extra compute for ~60% less activation memory.
        """
        h = self.input_proj(x)
        for i in range(len(self.convs)):
            h_res = h
            if self.training and self.use_checkpointing:
                # Checkpoint each GIN layer — recomputed during backward
                h = checkpoint(
                    self._gin_layer_fn, h, edge_index, i,
                    use_reentrant=False,
                )
            else:
                h = self._gin_layer_fn(h, edge_index, i)
            h = h + h_res
        return h

    def _gin_layer_fn(
        self, h: torch.Tensor, edge_index: torch.Tensor, layer_idx: int,
    ) -> torch.Tensor:
        """Single GIN layer (conv + norm + relu + dropout). Separated so
        ``torch.utils.checkpoint.checkpoint`` can wrap it."""
        h = self.convs[layer_idx](h, edge_index)
        h = self.norms[layer_idx](h)
        h = F.relu(h)
        h = self.dropout(h)
        return h

    @staticmethod
    def _scatter_log_softmax(
        values: torch.Tensor, batch: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised per-graph log-softmax (no Python for-loop).

        Equivalent to:
            for g in unique(batch):
                out[batch==g] = log_softmax(values[batch==g])
        but O(N) with scatter ops.
        """
        # Numerical stability: subtract max per graph
        max_per_graph = torch.zeros(
            batch.max().item() + 1, device=values.device, dtype=values.dtype,
        )
        max_per_graph.scatter_reduce_(
            0, batch, values, reduce='amax', include_self=False,
        )
        shifted = values - max_per_graph[batch]

        exp_shifted = shifted.exp()

        # Sum of exp per graph
        sum_exp = torch.zeros_like(max_per_graph)
        sum_exp.scatter_add_(0, batch, exp_shifted)

        log_probs = shifted - sum_exp[batch].log()
        return log_probs

    def forward(
        self,
        fcl_batch: Batch,
        h_mix: Optional[torch.Tensor] = None,
        p_iso: Optional[torch.Tensor] = None,
        h_dag: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            fcl_batch: Batched FCL graphs (x = 14-dim formula features).
            h_mix: [B, 256] — SMDG embeddings per feature.
            p_iso: [B, 9]  — isotope probabilities per feature.
            h_dag: [B, 128] — DAG-GNN fragmentation embeddings per feature (optional).
            return_hidden: If True, also return the final GIN hidden state h.

        Returns:
            energies:  [total_nodes] — raw energy scores per candidate.
            log_probs: [total_nodes] — log P(F_ij | feature) per candidate.
            h:         [total_nodes, hidden_dim] — (only if return_hidden=True)
        """
        x, edge_index, batch, formula_h = self._build_input(fcl_batch, h_mix, p_iso, h_dag)

        h = self._run_gin(x, edge_index)
        del x  # free large concatenated input

        # ── GIN energy pathway ──────────────────────────────────
        raw_energies = self.readout(h).squeeze(-1)  # [total_nodes]

        if self.use_bypass and formula_h is not None:
            gin_energies = raw_energies * self.energy_scale

            # ── Formula bypass pathway (immune to GIN smoothing) ────
            bypass_energy = self.formula_bypass(formula_h).squeeze(-1)  # [N]

            # ── Feature–formula interaction pathway ─────────────────
            if h_mix is not None:
                mix_h = self.mix_proj(h_mix[batch])      # [N, 128]
                interaction = (formula_h * mix_h).sum(-1)  # [N]
            else:
                interaction = torch.zeros_like(bypass_energy)

            energies = gin_energies + bypass_energy + interaction
        else:
            energies = raw_energies

        # Vectorised per-graph log-softmax (replaces Python for-loop)
        log_probs = self._scatter_log_softmax(-energies, batch)

        if return_hidden:
            return energies, log_probs, h
        return energies, log_probs

    def predict_probabilities(
        self,
        fcl_batch: Batch,
        h_mix: Optional[torch.Tensor] = None,
        p_iso: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return normalized probabilities P(F_ij) per candidate."""
        _, log_probs = self.forward(fcl_batch, h_mix, p_iso)
        return log_probs.exp()

    def apply_hard_constraints(
        self,
        energies: torch.Tensor,
        fcl_batch: Batch,
    ) -> torch.Tensor:
        """
        Zero out probabilities for physically invalid candidates.

        Checks: DBE ≥ 0, valence OK, ppm error ≤ 3.
        Assumes fcl_batch.x[:, 0] = DBE, [:, 8] = valence_ok, [:, -1] or stored ppm.
        """
        x = fcl_batch.x
        # formula features: [DBE, O/C, H/C, N/C, S/C, F/C, ppm_error, mass_exact, valence_ok, electron_parity]
        dbe = x[:, 0]
        ppm_error = x[:, 6]
        valence_ok = x[:, 8]

        # Mask: True = valid candidate
        valid = (dbe >= 0) & (valence_ok > 0.5) & (ppm_error.abs() <= 3.0)

        # Set energy of invalid candidates to very high (so exp(-E) → 0)
        constrained_energies = energies.clone()
        constrained_energies[~valid] = 1e6

        return constrained_energies
