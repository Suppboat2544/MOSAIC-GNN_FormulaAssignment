"""
P3-T2: FiLM Domain Conditioning & Gradient Reversal
=====================================================
Feature-wise Linear Modulation (FiLM) conditions the SMDG R-GCN
on the sample's matrix type (river DOM, estuary, reference, etc.).

    h^(l+1) = γ(z_matrix) ⊙ h_std^(l+1) + β(z_matrix)

A gradient reversal layer (GRL) is applied on top of node embeddings
to train a domain classifier that forces matrix-invariant features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional


# ── Gradient Reversal Layer ──────────────────────────────────

class _GradientReversal(Function):
    """Gradient reversal for domain-adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scaling factor alpha."""
    return _GradientReversal.apply(x, alpha)


# ── FiLM Layer ───────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Given a conditioning vector z ∈ R^{cond_dim}:
        gamma = Linear(z) + 1   (scale, initialised near identity)
        beta  = Linear(z)       (shift, initialised near zero)
        output = gamma ⊙ h + beta
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj = nn.Linear(cond_dim, hidden_dim)

        # Initialise near identity transform
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [N, hidden_dim] — node features
            z: [hidden_dim] or [1, hidden_dim] — conditioning vector
                (broadcast to all nodes)
        Returns:
            [N, hidden_dim] — modulated features
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        gamma = self.gamma_proj(z)   # [1, hidden_dim]
        beta = self.beta_proj(z)     # [1, hidden_dim]
        return gamma * h + beta


# ── Domain Classifier ────────────────────────────────────────

class DomainClassifier(nn.Module):
    """
    Domain classifier head with gradient reversal.

    Trained to predict matrix type from node embeddings.
    The GRL ensures the encoder learns matrix-invariant representations.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_domains: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_domains),
        )

    def forward(
        self,
        h: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            h: [N, hidden_dim] — node embeddings.
            alpha: GRL scaling (ramp up during training).

        Returns:
            logits: [N, num_domains] — domain prediction logits.
        """
        h_rev = gradient_reversal(h, alpha)
        return self.classifier(h_rev)


# ── Matrix Embedding ─────────────────────────────────────────

# Recognised matrix types  (extend as needed)
MATRIX_TYPES = [
    'river_DOM',       # Harney River samples
    'wetland_DOM',     # Pantanal samples
    'reference_FA',    # Suwannee / Pahokee fulvic acid
    'high_res_21T',    # 21 T instrument samples
    'PFAS_AFFF',       # PFAS AFFF matrices (21T FT-ICR)
    'marine_DOM',      # Open ocean / coastal DOM (PANGAEA)
    'arctic_DOM',      # Arctic river / Lena Delta DOM
    'estuarine_DOM',   # Estuarine / tidal DOM (BCO-DMO)
    'mixed_DOM',       # KNN / mixed provenance
    'unknown',         # fall-back
]
MATRIX_TO_IDX = {m: i for i, m in enumerate(MATRIX_TYPES)}
NUM_MATRIX_TYPES = len(MATRIX_TYPES)


def infer_matrix_type(sample_id: str) -> int:
    """Heuristic mapping of sample_id → matrix type index."""
    sid = sample_id.lower()
    if 'pfas' in sid or 'afff' in sid:
        return MATRIX_TO_IDX['PFAS_AFFF']
    if 'pantanal' in sid:
        return MATRIX_TO_IDX['wetland_DOM']
    if 'harney' in sid and '21t' in sid:
        return MATRIX_TO_IDX['high_res_21T']
    if 'harney' in sid:
        return MATRIX_TO_IDX['river_DOM']
    if 'suwannee' in sid and '21t' in sid:
        return MATRIX_TO_IDX['high_res_21T']
    if 'suwannee' in sid or 'pahokee' in sid or 'srfa' in sid or 'ppfa' in sid:
        return MATRIX_TO_IDX['reference_FA']
    # ── New environment types ──
    if 'knn' in sid:
        return MATRIX_TO_IDX['mixed_DOM']
    if any(k in sid for k in ['lena', 'arctic', 'l14_', 'pangaea_933']):
        return MATRIX_TO_IDX['arctic_DOM']
    if any(k in sid for k in ['he361', 'north_sea', 'osterholz', '848886']):
        return MATRIX_TO_IDX['marine_DOM']
    if any(k in sid for k in ['martinez', 'mediterranean', 'pangaea_887', 'ocean', 'atlantic', 'pangaea_969', 'moye', 'ps67']):
        return MATRIX_TO_IDX['marine_DOM']
    if any(k in sid for k in ['doboy', 'estuar', 'bcodmo']):
        return MATRIX_TO_IDX['estuarine_DOM']
    if any(k in sid for k in ['pangaea_962', 'global']):
        return MATRIX_TO_IDX['marine_DOM']
    if 'cram' in sid:
        return MATRIX_TO_IDX['marine_DOM']
    return MATRIX_TO_IDX['unknown']


class MatrixEmbedding(nn.Module):
    """Learnable embedding for sample matrix type."""

    def __init__(
        self,
        num_types: int = NUM_MATRIX_TYPES,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_types, embed_dim)

    def forward(self, matrix_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            matrix_idx: scalar or [B] — matrix type index.
        Returns:
            z: [embed_dim] or [B, embed_dim]
        """
        return self.embed(matrix_idx)
