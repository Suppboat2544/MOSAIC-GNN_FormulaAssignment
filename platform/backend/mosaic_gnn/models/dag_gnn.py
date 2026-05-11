"""
P3-T5: Fragmentation DAG-GNN — In-Source Neutral Loss Grammar
==============================================================
Directed Acyclic Graph GNN for modeling in-source fragmentation
patterns. Each precursor feature's fragments are organized into
a DAG where directed edges (parent → child) represent neutral losses.

Architecture:
  - Bottom-up (leaf → root) message passing using topological sort order
  - GRU-based recurrence for chain fragments
  - Root aggregation produces h_dag ∈ R^128 per precursor

Additional outputs:
  - Grammar violation score per candidate formula
  - Mass balance violation log per DAG edge

The h_dag embedding is concatenated to Module 3 GIN scorer input,
providing fragmentation context for formula ranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Neutral Loss Dictionary (NLD) — 12 entries
# ─────────────────────────────────────────────────────────────
NLD: Dict[str, float] = {
    'CO2':   43.98983,
    'HF':    20.00623,
    'H2O':   18.01056,
    'C2H4':  28.03130,
    'CH2':   14.01565,
    'SO3':   79.95682,
    'NH3':   17.02655,
    'CO':    27.99491,
    'COF2':  65.99185,
    'CF2':   49.99694,
    'HCl':   35.97668,
    'C2H2':  26.01565,
}

NLD_NAMES = list(NLD.keys())
NLD_MASSES = np.array(list(NLD.values()), dtype=np.float64)


def build_fragmentation_dag(
    precursor_mz: float,
    fragment_mz: np.ndarray,
    fragment_intensity: np.ndarray,
    ppm_error: float = 5.0,
    sn_threshold: float = 3.0,
) -> Data:
    """
    Construct a fragmentation DAG from precursor and in-source fragments.

    Root node = precursor ion.
    Child nodes = in-source fragment ions (m/z < precursor, I > S/N threshold).
    Directed edges from larger m/z → smaller m/z, labeled by neutral loss type.

    Args:
        precursor_mz:       Precursor ion m/z.
        fragment_mz:        Array of fragment m/z values.
        fragment_intensity: Array of fragment intensities.
        ppm_error:          Mass tolerance in ppm.
        sn_threshold:       Minimum S/N for fragment inclusion.

    Returns:
        PyG Data with DAG structure, node/edge features.
    """
    # Filter fragments below precursor and above S/N threshold
    valid = (fragment_mz < precursor_mz) & (fragment_intensity > sn_threshold)
    frag_mz = fragment_mz[valid]
    frag_int = fragment_intensity[valid]

    # Sort fragments by m/z descending (largest first, closer to precursor)
    sort_idx = np.argsort(-frag_mz)
    frag_mz = frag_mz[sort_idx]
    frag_int = frag_int[sort_idx]

    N_frags = len(frag_mz)

    # Node 0 = precursor, nodes 1..N_frags = fragments
    N_nodes = 1 + N_frags
    all_mz = np.concatenate([[precursor_mz], frag_mz])
    all_int = np.concatenate([[1.0], frag_int])  # precursor intensity = 1.0 (normalized)
    sigma = all_mz * ppm_error / 1e6

    # Node features: [m/z, log10(intensity+1), sigma, n_candidates, is_precursor]
    node_feat = np.zeros((N_nodes, 5), dtype=np.float32)
    node_feat[:, 0] = all_mz / 1000.0            # normalized m/z
    node_feat[:, 1] = np.log10(all_int + 1)       # log intensity
    node_feat[:, 2] = sigma                        # mass uncertainty
    node_feat[:, 3] = 0.0                          # candidate count (filled later)
    node_feat[:, 4] = np.array([1.0] + [0.0] * N_frags)  # is_precursor flag

    # Build edges: parent → child if neutral loss matches NLD
    edge_src = []
    edge_dst = []
    edge_feat_list = []

    for i in range(N_nodes):
        for j in range(N_nodes):
            if all_mz[i] <= all_mz[j]:
                continue  # parent must have larger m/z
            delta_m = all_mz[i] - all_mz[j]
            sigma_ij = np.sqrt(sigma[i]**2 + sigma[j]**2)
            tol_ij = 3 * sigma_ij

            for k, (nl_name, nl_mass) in enumerate(NLD.items()):
                residual = abs(delta_m - nl_mass)
                if residual < tol_ij:
                    # Mass balance residual
                    mb_residual = abs(all_mz[i] - all_mz[j] - nl_mass)
                    mb_valid = 1.0 if mb_residual < tol_ij else 0.0

                    # Edge features: [NL type one-hot (12), NL exact mass, residual ppm,
                    #                  mass balance residual, mass balance valid flag]
                    nl_onehot = np.zeros(len(NLD), dtype=np.float32)
                    nl_onehot[k] = 1.0
                    ppm_res = (delta_m - nl_mass) / nl_mass * 1e6

                    edge_f = np.concatenate([
                        nl_onehot,
                        [nl_mass / 100.0, ppm_res / 10.0, mb_residual, mb_valid],
                    ])
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_feat_list.append(edge_f)

    # Construct PyG Data
    x = torch.tensor(node_feat, dtype=torch.float32)

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat_list), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(NLD) + 4), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N_nodes,
    )
    data.precursor_mz = precursor_mz
    data.n_fragments = N_frags

    return data


def compute_grammar_violation_score(
    candidate_formula: Dict[str, int],
    observed_neutral_losses: List[str],
    candidate_mass: float,
) -> float:
    """
    Compute grammar violation score for a candidate formula.

    For each observed neutral loss, check if subtracting the loss formula
    from the candidate yields a valid residual (all counts ≥ 0, DBE ≥ 0).

    Score = fraction of observed losses that are incompatible with
    the candidate formula.

    Args:
        candidate_formula:     Element count dict, e.g. {'C': 10, 'H': 20, 'O': 5}.
        observed_neutral_losses: List of NLD keys observed in fragmentation.
        candidate_mass:        Exact mass of candidate.

    Returns:
        Grammar violation score in [0, 1]. 0 = fully compatible, 1 = fully incompatible.
    """
    if not observed_neutral_losses:
        return 0.0

    # Neutral loss formula element counts
    NL_COMPOSITIONS = {
        'CO2':  {'C': 1, 'O': 2},
        'HF':   {'H': 1, 'F': 1},
        'H2O':  {'H': 2, 'O': 1},
        'C2H4': {'C': 2, 'H': 4},
        'CH2':  {'C': 1, 'H': 2},
        'SO3':  {'S': 1, 'O': 3},
        'NH3':  {'N': 1, 'H': 3},
        'CO':   {'C': 1, 'O': 1},
        'COF2': {'C': 1, 'O': 1, 'F': 2},
        'CF2':  {'C': 1, 'F': 2},
        'HCl':  {'H': 1, 'Cl': 1},
        'C2H2': {'C': 2, 'H': 2},
    }

    violations = 0
    for nl_name in observed_neutral_losses:
        nl_comp = NL_COMPOSITIONS.get(nl_name, {})
        if not nl_comp:
            continue

        # Check: candidate - neutral_loss ≥ 0 for all elements
        valid = True
        residual_counts = dict(candidate_formula)
        for elem, n in nl_comp.items():
            remaining = residual_counts.get(elem, 0) - n
            if remaining < 0:
                valid = False
                break
            residual_counts[elem] = remaining

        if valid:
            # Additional check: residual must have DBE ≥ 0
            c = residual_counts.get('C', 0)
            h = residual_counts.get('H', 0)
            n = residual_counts.get('N', 0)
            # DBE = C - H/2 + N/2 + 1
            dbe = c - h / 2.0 + n / 2.0 + 1
            if dbe < 0:
                valid = False

        if not valid:
            violations += 1

    return violations / len(observed_neutral_losses)


class DAGGNNEncoder(nn.Module):
    """
    DAG-GNN encoder: bottom-up message passing through fragmentation DAG.

    Architecture:
      Layer 1: Leaf node embeddings via MLP(5 → 64)
      Layer 2: Parent aggregates children via MLP(parent || mean(children) || edge_feat)
      Layer 3: Root aggregation → h_dag = MLP(h_root → 128)

    Uses GRU-based recurrence for chain fragments:
      h_node = GRU(h_node, aggregated_children)

    Message passing follows topological sort order (leaves first, root last).
    """

    def __init__(
        self,
        node_feat_dim: int = 5,
        edge_feat_dim: int = 16,   # 12 NL onehot + 4
        hidden_dim: int = 64,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Leaf / initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Parent aggregation: MLP(h_parent || h_children_mean || h_edge_mean)
        self.aggregate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # GRU cell for recurrent message passing
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Root → output projection
        self.root_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def _topological_sort(self, edge_index: torch.Tensor, num_nodes: int) -> List[int]:
        """
        Topological sort of DAG nodes. Leaves come first, root last.
        Uses Kahn's algorithm (BFS-based).
        """
        if edge_index.numel() == 0:
            return list(range(num_nodes))

        # Compute in-degree
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        dst = edge_index[1]
        for d in dst:
            in_degree[d.item()] += 1

        # BFS from nodes with in_degree = 0 (these are the root/parent nodes)
        # But we want leaves first, so we reverse the edge direction logic
        # In our DAG: edges go parent(larger) → child(smaller)
        # So leaves have in_degree from reversed perspective = 0 out-degree
        out_degree = torch.zeros(num_nodes, dtype=torch.long)
        src = edge_index[0]
        for s in src:
            out_degree[s.item()] += 1

        # Leaves = nodes with out_degree = 0 (no children)
        queue = [i for i in range(num_nodes) if out_degree[i].item() == 0]
        order = []
        visited = set()

        # Reverse BFS: from leaves upward
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            order.append(node)

            # Find parents of this node (nodes that have edge → this node)
            parent_mask = dst == node
            if parent_mask.any():
                parents = src[parent_mask]
                for p in parents:
                    p_item = p.item()
                    if p_item not in visited:
                        # Check if all children of p are visited
                        children_of_p = dst[src == p_item]
                        if all(c.item() in visited for c in children_of_p):
                            queue.append(p_item)

        # Add any unvisited nodes (isolated nodes)
        for i in range(num_nodes):
            if i not in visited:
                order.append(i)

        return order

    def forward(self, dag_batch: Batch) -> torch.Tensor:
        """
        Bottom-up message passing through batched fragmentation DAGs.

        Args:
            dag_batch: Batched PyG Data of DAG graphs.

        Returns:
            h_dag: [B, out_dim] — root embedding per precursor.
        """
        x = dag_batch.x                    # [total_nodes, 5]
        edge_index = dag_batch.edge_index   # [2, total_edges]
        batch = dag_batch.batch             # [total_nodes]
        edge_attr = dag_batch.edge_attr if hasattr(dag_batch, 'edge_attr') else None

        device = x.device
        N = x.size(0)
        B = batch.max().item() + 1

        # Initial node embeddings
        h = self.node_encoder(x)  # [N, hidden_dim]

        # Edge embeddings (if available)
        if edge_attr is not None and edge_attr.numel() > 0:
            h_edge = self.edge_encoder(edge_attr)  # [E, hidden_dim]
        else:
            h_edge = None

        # Process each graph in the batch via topological sort
        # (This is done per-graph since topological order varies)
        for b in range(B):
            node_mask = batch == b
            node_indices = node_mask.nonzero(as_tuple=True)[0]
            if node_indices.numel() <= 1:
                continue  # single-node DAG, no message passing

            # Get edges within this graph
            if edge_index.numel() > 0:
                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                local_edges = edge_index[:, edge_mask]
            else:
                local_edges = torch.zeros((2, 0), dtype=torch.long, device=device)

            if local_edges.numel() == 0:
                continue

            # Topological sort (leaves first)
            order = self._topological_sort(local_edges, node_indices.numel())

            # Map local indices to global
            idx_map = {i: node_indices[i].item() for i in range(len(node_indices))}

            # Remap edges to local indices
            local_src = local_edges[0]
            local_dst = local_edges[1]

            # Bottom-up: process nodes in topological order (leaves → root)
            for local_idx in order:
                global_idx = idx_map.get(local_idx, local_idx)

                # Find children of this node (edges where this node is src)
                child_mask = local_src == global_idx
                if not child_mask.any():
                    continue  # leaf node — keep initial embedding

                child_global = local_dst[child_mask]

                # Aggregate children
                child_h = h[child_global]   # [n_children, hidden_dim]
                child_mean = child_h.mean(dim=0)  # [hidden_dim]

                # Aggregate edge features for edges from this node
                if h_edge is not None:
                    edge_indices_for_node = edge_mask.nonzero(as_tuple=True)[0][child_mask]
                    if edge_indices_for_node.numel() > 0:
                        edge_h_mean = h_edge[edge_indices_for_node].mean(dim=0)
                    else:
                        edge_h_mean = torch.zeros(self.hidden_dim, device=device)
                else:
                    edge_h_mean = torch.zeros(self.hidden_dim, device=device)

                # Aggregate: h_parent || h_children_mean || h_edge_mean
                combined = torch.cat([h[global_idx], child_mean, edge_h_mean])
                agg = self.aggregate_mlp(combined.unsqueeze(0)).squeeze(0)

                # GRU update
                h[global_idx] = self.gru(agg.unsqueeze(0), h[global_idx].unsqueeze(0)).squeeze(0)

        # Extract root node embedding per graph
        # Root = node with is_precursor flag = 1 (feature index 4)
        h_dag = torch.zeros(B, self.out_dim, device=device)

        for b in range(B):
            node_mask = batch == b
            node_indices = node_mask.nonzero(as_tuple=True)[0]
            if node_indices.numel() == 0:
                continue

            # Find root (is_precursor = 1)
            precursor_flag = x[node_indices, 4]
            root_mask = precursor_flag > 0.5
            if root_mask.any():
                root_idx = node_indices[root_mask][0]
            else:
                # Fallback: use first node
                root_idx = node_indices[0]

            h_dag[b] = self.root_proj(h[root_idx])

        return h_dag

    def forward_single(self, dag_data: Data) -> torch.Tensor:
        """
        Forward pass for a single (unbatched) DAG graph.

        Returns:
            h_dag: [1, out_dim] root embedding.
        """
        # Create a trivial batch
        dag_data.batch = torch.zeros(dag_data.num_nodes, dtype=torch.long,
                                      device=dag_data.x.device)
        batch = Batch.from_data_list([dag_data])
        return self.forward(batch)


def make_empty_dag() -> Data:
    """
    Create a single-node (precursor-only) DAG for features without fragments.

    The DAG-GNN encoder will output h_dag = zeros(128) for these.
    """
    x = torch.zeros(1, 5, dtype=torch.float32)
    x[0, 4] = 1.0  # is_precursor flag
    return Data(
        x=x,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, len(NLD) + 4), dtype=torch.float32),
        num_nodes=1,
    )
