"""
P3-T3: Evidential Deep Learning Uncertainty Quantification
============================================================
Normal-Inverse-Gamma (NIG) head that replaces the scalar energy
readout of the GIN scorer with four parameters (γ, ν, α, β)
describing a posterior predictive Student-t distribution.

Uncertainty decomposition:
    Aleatoric:  U_ale = β / (α − 1)
    Epistemic:  U_epi = β / (ν · (α − 1))

The evidential loss regularises evidence to correlate with
prediction error, yielding well-calibrated rejection curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvidentialHead(nn.Module):
    """
    Evidential regression head (Normal-Inverse-Gamma).

    Input:  h ∈ R^{N × hidden_dim}  (GIN final hidden states)
    Output: (gamma, nu, alpha, beta) each ∈ R^{N}

    Parameterisation:
        gamma = Linear(h)           — predicted mean (unconstrained)
        nu    = softplus(Linear(h)) — evidence for mean
        alpha = softplus(Linear(h)) + min_alpha — shape parameter
        beta  = softplus(Linear(h)) — rate parameter
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        min_alpha: float = 1.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.min_alpha = min_alpha

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Four separate output heads
        self.gamma_head = nn.Linear(128, 1)
        self.nu_head = nn.Linear(128, 1)
        self.alpha_head = nn.Linear(128, 1)
        self.beta_head = nn.Linear(128, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: [N, hidden_dim] — GIN hidden states per candidate.

        Returns:
            Dictionary with:
                gamma: [N]  — predicted mean (energy)
                nu:    [N]  — evidence for the mean
                alpha: [N]  — inverse-gamma shape (> min_alpha)
                beta:  [N]  — inverse-gamma rate (> 0)
                u_ale: [N]  — aleatoric uncertainty
                u_epi: [N]  — epistemic uncertainty
        """
        z = self.shared(h)

        gamma = self.gamma_head(z).squeeze(-1)
        nu = F.softplus(self.nu_head(z).squeeze(-1)) + 1e-6
        alpha = F.softplus(self.alpha_head(z).squeeze(-1)) + self.min_alpha
        beta = F.softplus(self.beta_head(z).squeeze(-1)) + 1e-6

        # Uncertainty decomposition
        u_ale = beta / (alpha - 1.0).clamp(min=1e-6)
        u_epi = beta / (nu * (alpha - 1.0).clamp(min=1e-6))

        return {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'u_ale': u_ale,
            'u_epi': u_epi,
        }

    def predict_with_uncertainty(
        self, h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.

        Returns:
            energies: [N] — predicted energies (gamma)
            u_ale:    [N] — aleatoric uncertainty
            u_epi:    [N] — epistemic uncertainty
        """
        out = self.forward(h)
        return out['gamma'], out['u_ale'], out['u_epi']
