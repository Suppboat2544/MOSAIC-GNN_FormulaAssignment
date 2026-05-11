"""
P3-T1: Semi-Supervised Masked Node Pretraining
================================================
Randomly masks 30 % of nodes in the SMDG graph, trains the R-GCN
encoder to reconstruct the elemental class from unmasked neighbours.

Loss:
    L_pretrain = L_mask_CE + μ · L_smooth

where L_smooth = Σ_{(i,j)∈E} w_ij · ‖h_i − h_j‖²
encourages smooth embeddings along known ETD edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MaskedNodePretrainer(nn.Module):
    """
    Self-supervised pretraining module for the SMDG R-GCN encoder.

    1) Mask 30 % of node features to zero.
    2) Run the SMDG encoder on the corrupted graph.
    3) Predict elemental class at masked positions via a linear probe.
    4) Smoothness regulariser over SMDG edges.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_elemental_classes: int = 7,
        mask_ratio: float = 0.30,
        smoothness_mu: float = 0.01,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.smoothness_mu = smoothness_mu

        # Linear probe on top of encoder output
        self.probe = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_elemental_classes),
        )

    def _mask_nodes(
        self, data: HeteroData,
    ) -> Tuple[HeteroData, torch.Tensor]:
        """
        Create a masked copy of the SMDG graph.

        Returns:
            masked_data: HeteroData with some node features zeroed.
            mask: bool tensor [N] — True at masked positions.
        """
        x = data['feature'].x.clone()
        N = x.size(0)
        n_mask = max(1, int(N * self.mask_ratio))

        perm = torch.randperm(N, device=x.device)
        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        mask[perm[:n_mask]] = True

        # Zero out masked nodes
        x[mask] = 0.0

        # Build a new HeteroData with masked x
        masked_data = HeteroData()
        masked_data['feature'].x = x
        for et in data.edge_types:
            masked_data[et].edge_index = data[et].edge_index
        return masked_data, mask

    def _smoothness_loss(
        self,
        h: torch.Tensor,
        data: HeteroData,
    ) -> torch.Tensor:
        """
        Edge smoothness regulariser:
            L_smooth = Σ_{(i,j)∈E} ‖h_i − h_j‖²  (mean over edges).
        """
        total = torch.tensor(0.0, device=h.device)
        count = 0
        for et in data.edge_types:
            ei = data[et].edge_index
            if ei.numel() == 0:
                continue
            src, dst = ei[0], ei[1]
            diff = h[src] - h[dst]
            total = total + (diff ** 2).sum(dim=-1).mean()
            count += 1
        if count == 0:
            return total
        return total / count

    def forward(
        self,
        encoder: nn.Module,
        data: HeteroData,
        eclass_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        One pretraining step.

        Args:
            encoder:  SMDGEncoder — the module being pretrained.
            data:     Original (un-masked) SMDG HeteroData.
            eclass_labels: [N] — elemental class ground-truth per node.

        Returns:
            loss: scalar — total pretraining loss.
            info: dict with loss breakdown.
        """
        masked_data, mask = self._mask_nodes(data)
        h = encoder(masked_data)               # [N, hidden_dim]

        # Classification at masked positions
        logits_masked = self.probe(h[mask])     # [n_mask, 7]
        targets = eclass_labels[mask][:logits_masked.size(0)]

        L_mask_CE = F.cross_entropy(logits_masked, targets)

        # Smoothness
        L_smooth = self._smoothness_loss(h, data)

        loss = L_mask_CE + self.smoothness_mu * L_smooth

        info = {
            'L_mask_CE': L_mask_CE.item(),
            'L_smooth': L_smooth.item(),
            'L_pretrain': loss.item(),
            'mask_acc': (logits_masked.argmax(-1) == targets).float().mean().item(),
        }
        return loss, info
