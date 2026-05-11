"""
P3-T6: IMS/CCS 3-D Joint Scoring
==================================
SchNet-inspired CCS predictor for collision cross section estimation
from molecular formula, enabling isobaric disambiguation when IMS
(Ion Mobility Spectrometry) data is available.

Joint scoring combines mass-based energy with CCS-based energy:
  E_joint = E_mass + α · |CCS_pred(F) - CCS_obs|

where α is a learnable parameter that weights the CCS contribution.
When has_ims=False, α is set to 0 (no CCS influence).

The CCS predictor uses a simplified SchNet architecture that operates
on molecular formula-level descriptors (not full 3D conformers) for
fast inference compatible with MS1-only workflow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SchNetCCSPredictor(nn.Module):
    """
    SchNet-inspired CCS predictor from molecular formula descriptors.

    Takes formula-level features (element counts, DBE, H/C, O/C, etc.)
    and predicts CCS (Å²) using a continuous-filter convolutional
    architecture adapted for tabular molecular descriptors.

    Simplified from full SE(3)-equivariant SchNet since we operate on
    formula-level features (not 3D coordinates) for MS1-only compatibility.

    Architecture:
        Input: [C, H, O, N, S, P, F, Cl, Br, exact_mass, DBE, H/C, O/C] → 13-dim
        Encoder: 3 SchNet-style interaction blocks
        Output: scalar CCS prediction (Å²)
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Interaction blocks (SchNet-style)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_gaussians, cutoff, dropout)
            for _ in range(num_interactions)
        ])

        # Output head: predict CCS (Å²)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        formula_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict CCS from formula-level features.

        Args:
            formula_features: [N, input_dim] molecular formula descriptors.

        Returns:
            ccs_pred: [N] predicted CCS in Å².
        """
        h = self.input_proj(formula_features)  # [N, hidden_dim]

        for interaction in self.interactions:
            h = interaction(h)

        ccs = self.output_head(h).squeeze(-1)  # [N]

        # CCS must be positive
        ccs = F.softplus(ccs) + 50.0  # minimum ~50 Å² for small molecules

        return ccs


class SchNetInteraction(nn.Module):
    """
    SchNet-style interaction block for formula-level features.

    Adapted from the original SchNet architecture:
    - Original: uses radial basis functions on 3D distances
    - This version: uses self-interaction on feature embeddings
      with Gaussian expansion for mass-based "distance" features
    """

    def __init__(
        self,
        hidden_dim: int,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Continuous filter: generates filter weights from input
        self.filter_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Interaction MLP
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Self-interaction with residual connection."""
        # Generate continuous filter
        W = self.filter_net(h)  # [N, hidden_dim]

        # Element-wise interaction (self-filter)
        h_filtered = h * W

        # MLP update
        h_out = self.interaction(h_filtered)

        # Residual + normalization
        return self.norm(h + h_out)


class JointScorer(nn.Module):
    """
    Joint mass + CCS scoring for formula candidates.

    Combines energy-based mass scoring (from GIN Module 3) with
    CCS-based scoring when IMS data is available:

        E_joint = E_mass + α · |CCS_pred(F) - CCS_obs|

    α is a learnable parameter, initialized > 0 so CCS contributes
    meaningfully from the start. Set α=0 for has_ims=False samples.

    CCS soft filtering: instead of hard CCS threshold, penalize
    candidates whose predicted CCS deviates from observed CCS.
    """

    def __init__(
        self,
        ccs_input_dim: int = 13,
        ccs_hidden_dim: int = 128,
        alpha_init: float = 0.5,
    ):
        super().__init__()

        self.ccs_predictor = SchNetCCSPredictor(
            input_dim=ccs_input_dim,
            hidden_dim=ccs_hidden_dim,
        )

        # Learnable CCS weight
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(
        self,
        e_mass: torch.Tensor,
        formula_features: torch.Tensor,
        ccs_observed: Optional[torch.Tensor] = None,
        has_ims: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint energy scores.

        Args:
            e_mass:           [N] mass-based energy scores from GIN.
            formula_features: [N, 13] formula descriptors for CCS prediction.
            ccs_observed:     [N] observed CCS values (if IMS available).
            has_ims:          Whether this sample has IMS data.

        Returns:
            e_joint: [N] joint energy scores.
            ccs_pred: [N] predicted CCS values.
        """
        # Always predict CCS (useful for evaluation even without observed CCS)
        ccs_pred = self.ccs_predictor(formula_features)  # [N]

        if has_ims and ccs_observed is not None:
            # CCS mismatch penalty
            ccs_diff = (ccs_pred - ccs_observed).abs()

            # Use softplus(alpha) to ensure positive CCS weight
            alpha = F.softplus(self.alpha)

            e_joint = e_mass + alpha * ccs_diff
        else:
            e_joint = e_mass

        return e_joint, ccs_pred


class CCSLoss(nn.Module):
    """
    CCS prediction loss for training the SchNet CCS predictor.

    L_CCS = MSE(CCS_pred, CCS_obs) for has_ims=True samples only.

    Only computed on samples with IMS data available.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        ccs_pred: torch.Tensor,
        ccs_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ccs_pred:     [N] predicted CCS values.
            ccs_observed: [N] observed CCS values.
            mask:         [N] bool — which entries have valid CCS.

        Returns:
            Scalar MSE loss on valid entries.
        """
        if mask is not None:
            ccs_pred = ccs_pred[mask]
            ccs_observed = ccs_observed[mask]

        if ccs_pred.numel() == 0:
            return torch.tensor(0.0, device=ccs_pred.device, requires_grad=True)

        return F.mse_loss(ccs_pred, ccs_observed)


def build_ccs_features(
    candidate_counts: Dict[str, int],
    exact_mass: float,
    dbe: float,
) -> np.ndarray:
    """
    Build 13-dim CCS predictor input features from a candidate formula.

    Features: [C, H, O, N, S, P, F, Cl, Br, exact_mass_norm, DBE_norm, H/C, O/C]

    Args:
        candidate_counts: Element count dictionary.
        exact_mass:       Exact molecular mass.
        dbe:              Double bond equivalents.

    Returns:
        13-element float32 array.
    """
    c = candidate_counts.get('C', 0)
    h = candidate_counts.get('H', 0)
    o = candidate_counts.get('O', 0)
    n = candidate_counts.get('N', 0)
    s = candidate_counts.get('S', 0)

    return np.array([
        c / 50.0,
        h / 100.0,
        o / 30.0,
        n / 10.0,
        s / 5.0,
        candidate_counts.get('P', 0) / 5.0,
        candidate_counts.get('F', 0) / 30.0,
        candidate_counts.get('Cl', 0) / 10.0,
        candidate_counts.get('Br', 0) / 5.0,
        exact_mass / 1000.0,
        dbe / 30.0,
        h / max(c, 1),
        o / max(c, 1),
    ], dtype=np.float32)
