"""
P2-T1: Module 1 — SMDG R-GCN Encoder
======================================
Relational Graph Convolutional Network on the heterogeneous
Spectral Mass-Difference Graph.

Uses basis decomposition W_r = Σ_b  a_rb · V_b to keep parameter
count tractable across many edge types, plus residual connections,
LayerNorm, and per-edge-type neighbor normalization.

Output: h_mix ∈ R^{N × 256} — mixture-context embedding per node.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from typing import Dict, List, Optional, Tuple
import math


# ─────────────────────────────────────────────────────────────
# Basis-Decomposed R-GCN Layer
# ─────────────────────────────────────────────────────────────

class BasisRGCNConv(MessagePassing):
    """
    Single R-GCN message-passing layer with basis decomposition.

    W_r = Σ_{b=1}^{B} a_{r,b} · V_b

    where V_b ∈ R^{in × out} are shared bases and a_{r,b} are
    learnable scalar coefficients per relation r.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 4,
    ):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Shared basis matrices V_b ∈ R^{in × out}
        self.bases = nn.Parameter(
            torch.empty(self.num_bases, in_channels, out_channels)
        )
        # Per-relation coefficients a_{r,b}
        self.att = nn.Parameter(
            torch.empty(num_relations, self.num_bases)
        )
        # Self-loop transform
        self.root = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[str, torch.Tensor],
        edge_type_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels].
            edge_index_dict: {edge_type_name: edge_index [2, E_r]}.
            edge_type_to_idx: {edge_type_name: int index r}.

        Returns:
            Updated node features [N, out_channels].
        """
        # Precompute W_r = Σ_b a_{r,b} · V_b  →  [R, in, out]
        # att: [R, B], bases: [B, in, out]
        W = torch.einsum('rb,bio->rio', self.att, self.bases)  # [R, in, out]

        out = self.root(x)  # self-loop

        for etype, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            r = edge_type_to_idx[etype]
            if r >= self.num_relations:
                continue  # skip edge types beyond model capacity
            W_r = W[r]  # [in, out]

            # Message: W_r · x_j, aggregated for node i
            x_j = x[edge_index[0]]           # source node features
            msg = x_j @ W_r                   # [E, out]

            # Neighbor normalization: divide by |N_r(i)|
            # Count neighbors per node for this relation
            N = x.size(0)
            deg = torch.zeros(N, device=x.device)
            deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
            deg_inv = (deg + 1e-8).pow(-1)

            # Scatter-add messages to target nodes
            agg = torch.zeros(N, self.out_channels, device=x.device)
            agg.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, self.out_channels), msg)

            # Normalize
            agg = agg * deg_inv.unsqueeze(1)
            out = out + agg

        return out + self.bias


# ─────────────────────────────────────────────────────────────
# Multi-Layer R-GCN Encoder
# ─────────────────────────────────────────────────────────────

class SMDGEncoder(nn.Module):
    """
    SMDG R-GCN Encoder (Module 1).

    4-layer R-GCN with basis decomposition, residual connections,
    LayerNorm, and dropout.

    Input : HeteroData with data['feature'].x ∈ R^{N × 4}
            (m/z, log10(int+1), sigma, charge_state)
    Output: h_mix ∈ R^{N × hidden_dim}
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_bases: int = 4,
        num_relations: int = 11,   # max possible ETD types
        dropout: float = 0.1,
        activation: str = 'ReLU',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # R-GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                BasisRGCNConv(hidden_dim, hidden_dim, num_relations, num_bases)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'ELU':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        self.use_checkpointing = True

    def forward(
        self,
        data: HeteroData,
        edge_type_to_idx: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            data: HeteroData with data['feature'].x and typed edge indices.
            edge_type_to_idx: Mapping from edge type name to integer index.
                             If None, auto-built from data.edge_types.

        Returns:
            h_mix: Tensor [N, hidden_dim] — mixture-context embeddings.
        """
        x = data['feature'].x  # [N, 4]
        h = self.input_proj(x)  # [N, hidden_dim]

        # Build edge_index_dict and edge_type_to_idx
        edge_index_dict = {}
        if edge_type_to_idx is None:
            edge_type_to_idx = {}
            idx = 0
            for et in data.edge_types:
                etype = et[1]  # ('feature', etype, 'feature')
                if etype not in edge_type_to_idx:
                    edge_type_to_idx[etype] = idx
                    idx += 1
                edge_index_dict[etype] = data[et].edge_index
        else:
            for et in data.edge_types:
                etype = et[1]
                if etype in edge_type_to_idx:
                    edge_index_dict[etype] = data[et].edge_index

        # Message-passing layers with residual + LayerNorm
        for i in range(self.num_layers):
            h_res = h
            if self.training and self.use_checkpointing:
                h = ckpt_fn(
                    self._rgcn_layer_fn, h, edge_index_dict, edge_type_to_idx, i,
                    use_reentrant=False,
                )
            else:
                h = self._rgcn_layer_fn(h, edge_index_dict, edge_type_to_idx, i)
            h = h + h_res  # Residual connection

        return h

    def _rgcn_layer_fn(self, h, edge_index_dict, edge_type_to_idx, layer_idx):
        """One R-GCN layer block (conv + norm + act + dropout)."""
        h = self.convs[layer_idx](h, edge_index_dict, edge_type_to_idx)
        h = self.norms[layer_idx](h)
        h = self.act(h)
        h = self.dropout(h)
        return h

    def count_parameters(self) -> Dict[str, int]:
        """Report parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        conv_params = sum(p.numel() for conv in self.convs for p in conv.parameters())
        proj_params = sum(p.numel() for p in self.input_proj.parameters())
        norm_params = sum(p.numel() for n in self.norms for p in n.parameters())

        return {
            'total': total,
            'conv_layers': conv_params,
            'input_projection': proj_params,
            'layer_norms': norm_params,
        }

    def compute_full_w_params(self, num_relations: int) -> int:
        """
        Compute how many params a full-weight R-GCN would use
        (without basis decomposition) for comparison.

        Full: num_relations × hidden_dim × hidden_dim per layer.
        """
        return self.num_layers * num_relations * self.hidden_dim * self.hidden_dim
