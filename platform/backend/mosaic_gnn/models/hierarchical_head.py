"""
P2-T5: Module 5 — Hierarchical Classification Head
====================================================
Three-level classification head for coarse-to-fine formula assignment:

  Level 1: Elemental class prediction (7 classes: CHO/CHON/CHOS/CHONS/CHOF/CHONSP/other)
  Level 2: DBE bin prediction (4 bins: 0-3, 4-7, 8-12, 13+), conditioned on L1
  Level 3: Specific formula ranking (GIN energy scores), masked by L1 & L2

During inference the hierarchy acts as a progressive filter:
  L1 mask → L2 mask → L3 re-rank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

ELEMENTAL_CLASSES = ['CHO', 'CHON', 'CHOS', 'CHONS', 'CHOF', 'CHONSP', 'other']
DBE_BIN_EDGES = [0, 4, 8, 13]  # bins: [0,3], [4,7], [8,12], [13+]
NUM_ELEMENTAL_CLASSES = len(ELEMENTAL_CLASSES)
NUM_DBE_BINS = 4

# Map elemental class name to index
ECLASS_TO_IDX = {c: i for i, c in enumerate(ELEMENTAL_CLASSES)}


def dbe_to_bin(dbe: float) -> int:
    """Convert a DBE value to a bin index (0-3)."""
    if dbe < 4:
        return 0
    elif dbe < 8:
        return 1
    elif dbe < 13:
        return 2
    else:
        return 3


class HierarchicalHead(nn.Module):
    """
    Module 5: Hierarchical Classification Head.

    Input:  GIN node embedding h ∈ R^{hidden_dim} per candidate
    Output: (L1 class logits, L2 DBE bin logits, masked L3 energy scores)

    Level 1: Linear(hidden_dim → 7) → softmax
    Level 2: Linear(hidden_dim + 7 → 4) → softmax
    Level 3: GIN energy scores masked by L1/L2 predictions
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_elemental_classes: int = NUM_ELEMENTAL_CLASSES,
        num_dbe_bins: int = NUM_DBE_BINS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_elemental_classes = num_elemental_classes
        self.num_dbe_bins = num_dbe_bins

        # Normalize GIN hidden states before classification
        # h_cand norms can be ~1000+ due to residual GIN accumulation;
        # LayerNorm stabilizes logit magnitudes across training.
        self.h_norm = nn.LayerNorm(hidden_dim)

        # Level 1: Elemental class classifier
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_elemental_classes),
        )

        # Level 2: DBE bin classifier (conditioned on L1)
        self.l2_head = nn.Sequential(
            nn.Linear(hidden_dim + num_elemental_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_dbe_bins),
        )

    def forward(
        self,
        h: torch.Tensor,
        candidate_eclass: Optional[torch.Tensor] = None,
        candidate_dbe_bin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical head.

        Args:
            h: [total_candidates, hidden_dim] — GIN embeddings per candidate.
            candidate_eclass: [total_candidates] — ground-truth elemental class (for teacher forcing).
            candidate_dbe_bin: [total_candidates] — ground-truth DBE bin (for teacher forcing).

        Returns:
            l1_logits: [total_candidates, 7] — elemental class logits.
            l2_logits: [total_candidates, 4] — DBE bin logits.
        """
        # Normalize hidden states for stable logits
        h = self.h_norm(h)

        # Level 1
        l1_logits = self.l1_head(h)  # [N, 7]

        # Level 2: condition on L1 output (soft one-hot)
        if self.training and candidate_eclass is not None:
            # Teacher forcing: use ground-truth L1 as one-hot
            l1_onehot = F.one_hot(candidate_eclass.long(), self.num_elemental_classes).float()
        else:
            # Inference: use predicted L1 probabilities
            l1_onehot = F.softmax(l1_logits, dim=-1)

        l2_input = torch.cat([h, l1_onehot], dim=-1)  # [N, hidden_dim + 7]
        l2_logits = self.l2_head(l2_input)  # [N, 4]

        return l1_logits, l2_logits

    def apply_hierarchical_mask(
        self,
        energies: torch.Tensor,
        l1_logits: torch.Tensor,
        l2_logits: torch.Tensor,
        candidate_eclass_labels: torch.Tensor,
        candidate_dbe_bins: torch.Tensor,
        batch: torch.Tensor,
        threshold_l1: float = 0.1,
        threshold_l2: float = 0.1,
    ) -> torch.Tensor:
        """
        Apply hierarchical mask at inference time.

        For each feature:
          1. Get L1 predicted class → mask out candidates with wrong elemental class
          2. Get L2 predicted DBE bin → mask out candidates with wrong DBE bin
          3. Remaining candidates ranked by L3 energy scores

        Args:
            energies:              [total_candidates] — GIN energy scores.
            l1_logits:             [total_candidates, 7] — elemental class logits.
            l2_logits:             [total_candidates, 4] — DBE bin logits.
            candidate_eclass_labels: [total_candidates] — elemental class index per candidate.
            candidate_dbe_bins:    [total_candidates] — DBE bin index per candidate.
            batch:                 [total_candidates] — feature assignment.
            threshold_l1:          Minimum L1 probability to keep a class.
            threshold_l2:          Minimum L2 probability to keep a DBE bin.

        Returns:
            masked_energies: [total_candidates] — energies with masked-out set to +inf.
        """
        masked_energies = energies.clone()

        # Per-feature masking
        num_features = batch.max().item() + 1
        for f_idx in range(num_features):
            mask = (batch == f_idx)
            if mask.sum() == 0:
                continue

            # L1: predicted elemental class (feature-level vote = mean over candidates)
            l1_probs = F.softmax(l1_logits[mask], dim=-1).mean(dim=0)
            predicted_l1 = l1_probs.argmax().item()

            # Mask candidates whose elemental class doesn't match predicted
            eclass = candidate_eclass_labels[mask]
            l1_valid = (eclass == predicted_l1)

            # Also keep classes with probability above threshold
            for cls_idx in range(self.num_elemental_classes):
                if l1_probs[cls_idx] >= threshold_l1:
                    l1_valid = l1_valid | (eclass == cls_idx)

            # L2: predicted DBE bin
            l2_probs = F.softmax(l2_logits[mask], dim=-1).mean(dim=0)
            predicted_l2 = l2_probs.argmax().item()

            dbe_bins = candidate_dbe_bins[mask]
            l2_valid = (dbe_bins == predicted_l2)
            for bin_idx in range(self.num_dbe_bins):
                if l2_probs[bin_idx] >= threshold_l2:
                    l2_valid = l2_valid | (dbe_bins == bin_idx)

            # Combined mask
            valid = l1_valid & l2_valid
            if valid.sum() == 0:
                # Fallback: keep all candidates if mask eliminates everything
                valid = torch.ones_like(l1_valid)

            # Apply mask to energies
            feature_energies = masked_energies[mask]
            feature_energies[~valid] = float('inf')
            masked_energies[mask] = feature_energies

        return masked_energies
