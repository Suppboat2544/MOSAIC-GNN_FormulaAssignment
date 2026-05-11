"""
MOSAIC-GNN Loss Functions
=========================
Implements the full multi-term loss:

  L_total = L_CE1 + L_CE2 + L_EBM + λ₁·L_ADC + λ₂·L_ISO + λ₃·L_UQ
            + λ_dag·L_DAG + λ_gram·L_grammar + λ_phys·L_physics + λ_ccs·L_CCS

where:
  L_CE1     : CrossEntropy on elemental class (inverse-frequency weighted)
  L_CE2     : CrossEntropy on DBE bin
  L_EBM     : InfoNCE contrastive loss (correct formula = positive,
               10 hard + 10 random negatives per feature)
  L_ADC     : Adduct consistency MSE (λ₁ = 0.1 → 0.5 annealed by epoch 20)
  L_ISO     : Roepstorff physics MSE (λ₂ = 0.2)
  L_UQ      : Evidential Normal-Inverse-Gamma NLL (λ₃ = 0.05)
  L_DAG     : CrossEntropy on DAG root formula prediction (λ_dag = 0.15)
  L_grammar : Mean grammar violation penalty (λ_gram = 0.05)
  L_physics : PINN mass conservation MSE (λ_phys = 1.0)
  L_CCS     : CCS prediction MSE when IMS data available (λ_ccs = 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ═══════════════════════════════════════════════════════════════
#  L_CE1: Elemental Class Cross-Entropy (inverse-frequency weighted)
# ═══════════════════════════════════════════════════════════════

class ElementalClassCE(nn.Module):
    """
    Weighted cross-entropy on the 7 elemental classes.
    Weights are computed as inverse-frequency to handle class imbalance:
      w_c = N_total / (C · N_c)
    """

    def __init__(self, num_classes: int = 7, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.register_buffer('class_weights', torch.ones(num_classes))

    def update_weights(self, labels: torch.Tensor):
        """Compute inverse-frequency weights from a batch of labels.
        
        Weights are capped to prevent extreme values for rare/empty classes:
          - Classes with 0 samples get weight=0 (can't learn them)
          - All weights capped at max 10.0
        """
        raw_counts = torch.bincount(labels, minlength=self.num_classes).float()
        empty_mask = raw_counts == 0  # classes with zero samples
        counts = raw_counts.clamp(min=1.0)
        total = counts.sum()
        weights = total / (self.num_classes * counts)
        weights = weights.clamp(max=10.0)  # cap extreme weights
        weights[empty_mask] = 0.0  # zero-out truly empty classes
        self.class_weights = weights.to(labels.device)

    def forward(
        self,
        logits: torch.Tensor,          # [N, 7]
        targets: torch.Tensor,         # [N] long
        update_weights: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [N, num_classes] raw class logits
            targets: [N] integer class labels
        Returns:
            Scalar cross-entropy loss
        """
        if update_weights:
            self.update_weights(targets)

        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights.to(logits.device),
            reduction='mean',
            label_smoothing=self.label_smoothing,
        )


# ═══════════════════════════════════════════════════════════════
#  L_CE2: DBE Bin Cross-Entropy
# ═══════════════════════════════════════════════════════════════

class DBEBinCE(nn.Module):
    """
    Cross-entropy on the 4 DBE bins: [0-3], [4-7], [8-12], [13+].
    """

    def __init__(self, num_bins: int = 4, label_smoothing: float = 0.0):
        super().__init__()
        self.num_bins = num_bins
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,   # [N, 4]
        targets: torch.Tensor,  # [N] long
    ) -> torch.Tensor:
        return F.cross_entropy(logits, targets, reduction='mean',
                               label_smoothing=self.label_smoothing)


# ═══════════════════════════════════════════════════════════════
#  L_EBM: InfoNCE Contrastive Loss
# ═══════════════════════════════════════════════════════════════

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for Energy-Based Model formula scoring.

    For each feature, the correct formula is the positive example.
    Negatives are sampled from:
      - num_hard_negatives: lattice neighbors of the correct formula
        (requires edge_index; falls back to random if unavailable)
      - num_random_negatives: random wrong candidates from the same feature

    In the EBM framework, lower energy = higher probability = better rank.
    We convert energies to scores via negation:  score = -energy
    so that InfoNCE correctly maximises the score of the correct formula.

    L_EBM = -log( exp(s_pos/τ) / [exp(s_pos/τ) + Σ exp(s_neg/τ)] )

    This is the standard InfoNCE / N-pair loss.
    """

    def __init__(
        self,
        num_hard_negatives: int = 10,
        num_random_negatives: int = 10,
        temperature: float = 1.0,
        max_loss: float = 50.0,
    ):
        super().__init__()
        self.num_hard_neg = num_hard_negatives
        self.num_random_neg = num_random_negatives
        self.temperature = temperature
        self.max_loss = max_loss

    def forward(
        self,
        energies: torch.Tensor,            # [total_candidates]
        candidate_to_feature: torch.Tensor, # [total_candidates] — maps candidate → feature
        correct_mask: torch.Tensor,         # [total_candidates] bool — True = correct formula
        edge_index: Optional[torch.Tensor] = None,  # [2, E] FCL lattice edges
    ) -> torch.Tensor:
        """
        Vectorised InfoNCE loss.

        For each feature that has a correct formula:
          1. Positive = score of correct candidate (score = -energy)
          2. Hard negatives = wrong lattice-neighbors of correct (via edge_index)
          3. Random negatives = random wrong candidates from same feature
          4. Compute -log softmax(s_pos) over [pos, hard_neg, random_neg]
        """
        device = energies.device

        # ── Convert energies to scores: lower energy = higher score ──
        scores = -energies

        # Identify features that have at least one correct AND one wrong candidate
        correct_indices = correct_mask.nonzero(as_tuple=True)[0]
        if correct_indices.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Features that have correct formulas
        correct_features = candidate_to_feature[correct_indices]
        unique_features = correct_features.unique()

        # For each such feature, take the FIRST correct candidate as positive
        # Use scatter to find earliest correct index per feature
        feat_to_pos_idx = torch.full(
            (candidate_to_feature.max().item() + 1,), -1,
            dtype=torch.long, device=device,
        )
        # Reverse iterate so earlier indices overwrite later ones
        rev_order = correct_indices.flip(0)
        rev_feats = candidate_to_feature[rev_order]
        feat_to_pos_idx.scatter_(0, rev_feats, rev_order)

        pos_indices = feat_to_pos_idx[unique_features]
        valid = pos_indices >= 0
        unique_features = unique_features[valid]
        pos_indices = pos_indices[valid]

        if unique_features.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        s_pos = scores[pos_indices]  # [F] — scores of correct formulas

        # For negatives: gather wrong candidates per feature
        wrong_mask = ~correct_mask
        wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
        wrong_features = candidate_to_feature[wrong_indices]

        # ── Pre-build hard-negative adjacency from lattice edges ──
        # For each positive candidate, find its lattice neighbors that are wrong
        hard_neg_map = {}  # pos_candidate_idx → tensor of wrong neighbor indices
        if edge_index is not None and edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            for i, pos_idx in enumerate(pos_indices):
                pi = pos_idx.item()
                feat_idx = unique_features[i].item()
                # Lattice neighbors of the positive
                neighbor_mask = (src == pi)
                if neighbor_mask.any():
                    neighbors = dst[neighbor_mask]
                    # Keep only wrong neighbors in the same feature
                    same_feat = candidate_to_feature[neighbors] == feat_idx
                    wrong_neigh = neighbors[same_feat & (~correct_mask[neighbors])]
                    if wrong_neigh.numel() > 0:
                        hard_neg_map[i] = wrong_neigh

        # ── Vectorized per-feature adaptive temperature ──
        # Compute per-feature score std using scatter ops (O(1) GPU calls),
        # NOT per-feature std() in a Python loop (O(233K) GPU syncs).
        # This preserves per-feature discrimination (run8: 92.34% Top-1)
        # while avoiding the 19x slowdown of the loop approach.
        # Detach so gradients flow through scores, not the scale factor.
        n_all_feats = candidate_to_feature.max().item() + 1
        _scores_det = scores.detach()
        _ones = torch.ones_like(_scores_det)

        feat_count = torch.zeros(n_all_feats, device=device)
        feat_count.scatter_add_(0, candidate_to_feature, _ones)

        feat_sum = torch.zeros(n_all_feats, device=device)
        feat_sum.scatter_add_(0, candidate_to_feature, _scores_det)
        feat_mean = feat_sum / feat_count.clamp(min=1)

        sq_diff = (_scores_det - feat_mean[candidate_to_feature]) ** 2
        feat_var = torch.zeros(n_all_feats, device=device)
        feat_var.scatter_add_(0, candidate_to_feature, sq_diff)
        feat_std = (feat_var / feat_count.clamp(min=1)).sqrt()

        # Per-feature temperature scale: clamp std ≥ 1 to prevent collapse
        feat_scale = feat_std.clamp(min=1.0) / self.temperature   # [n_all_feats]

        # ── Build per-feature negative sampling via grouping ──
        losses = []
        for i, feat_idx in enumerate(unique_features):
            feat_wrong = wrong_indices[wrong_features == feat_idx]
            if feat_wrong.numel() == 0:
                continue

            neg_indices_list = []

            # Hard negatives: lattice neighbors of positive (if available)
            if i in hard_neg_map:
                hard_pool = hard_neg_map[i]
                n_hard = min(self.num_hard_neg, hard_pool.size(0))
                hard_perm = torch.randperm(hard_pool.size(0), device=device)[:n_hard]
                neg_indices_list.append(hard_pool[hard_perm])

            # Random negatives: sample from all wrong candidates of this feature
            n_random_target = self.num_random_neg
            if i not in hard_neg_map:
                # No hard negatives available — use full budget for random
                n_random_target = self.num_hard_neg + self.num_random_neg
            n_random = min(n_random_target, feat_wrong.size(0))
            perm = torch.randperm(feat_wrong.size(0), device=device)[:n_random]
            neg_indices_list.append(feat_wrong[perm])

            # Concatenate and deduplicate
            all_neg_idx = torch.cat(neg_indices_list).unique()

            s_neg = scores[all_neg_idx]
            scale_i = feat_scale[feat_idx]
            all_s = torch.cat([s_pos[i:i+1], s_neg]) / scale_i
            log_softmax_val = all_s[0] - torch.logsumexp(all_s, dim=0)
            losses.append((-log_softmax_val).clamp(max=self.max_loss))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()



# ═══════════════════════════════════════════════════════════════
#  L_ISO: Roepstorff Isotope Physics MSE
# ═══════════════════════════════════════════════════════════════

class IsotopePhysicsLoss(nn.Module):
    """
    MSE between predicted (Roepstorff-theoretical) isotope ratios
    and observed M+1/M+2 ratios.

    L_ISO = 0.5 * [MSE(theo_m1, obs_m1) + MSE(theo_m2, obs_m2)]
    """

    def forward(
        self,
        theo_m1: torch.Tensor,       # [N_iso] predicted M+1 ratio
        theo_m2: torch.Tensor,       # [N_iso] predicted M+2 ratio
        obs_m1: torch.Tensor,        # [N_iso] observed M+1 ratio
        obs_m2: torch.Tensor,        # [N_iso] observed M+2 ratio
        mask: Optional[torch.Tensor] = None,  # [N_iso] bool — valid entries
    ) -> torch.Tensor:
        if mask is not None:
            theo_m1 = theo_m1[mask]
            theo_m2 = theo_m2[mask]
            obs_m1 = obs_m1[mask]
            obs_m2 = obs_m2[mask]

        if theo_m1.numel() == 0:
            return torch.tensor(0.0, device=theo_m1.device, requires_grad=True)

        loss_m1 = F.mse_loss(theo_m1, obs_m1)
        loss_m2 = F.mse_loss(theo_m2, obs_m2)
        raw = 0.5 * (loss_m1 + loss_m2)
        return raw.clamp(max=10.0)  # cap to prevent L_ISO dominating


# ═══════════════════════════════════════════════════════════════
#  L_UQ: Evidential Normal-Inverse-Gamma NLL
# ═══════════════════════════════════════════════════════════════

class EvidentialNIGLoss(nn.Module):
    """
    Evidential regression loss using Normal-Inverse-Gamma (NIG) prior.

    Each candidate's energy prediction is modeled as:
        y ~ N(μ, σ²)  with  σ² ~ InvGamma(α, β)

    The NIG negative log-likelihood is:
        NLL = 0.5·log(π/ν) - α·log(Ω) + (α + 0.5)·log((y-γ)²·ν + Ω)
              + log(Γ(α)/Γ(α+0.5))

    where γ = predicted mean, ν = virtual evidence count, α = shape, β = rate,
    and Ω = 2β(1+ν).

    Additionally, a regularizer penalizes evidence on incorrect predictions:
        L_reg = |y - γ| · (2ν + α)

    Reference: Amini et al., "Deep Evidential Regression" (NeurIPS 2020).
    """

    def __init__(self, coeff: float = 0.05, min_alpha: float = 1.01):
        super().__init__()
        self.coeff = coeff
        self.min_alpha = min_alpha

    def forward(
        self,
        gamma: torch.Tensor,   # [N] predicted mean (formula energy)
        nu: torch.Tensor,      # [N] virtual evidence count (> 0)
        alpha: torch.Tensor,   # [N] shape parameter (> 1)
        beta: torch.Tensor,    # [N] rate parameter (> 0)
        target: torch.Tensor,  # [N] observed target value
    ) -> torch.Tensor:
        """
        Returns scalar NIG loss averaged over all samples.
        """
        # Enforce parameter constraints
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + self.min_alpha
        beta = F.softplus(beta) + 1e-6

        # Omega
        Omega = 2.0 * beta * (1.0 + nu)

        # NLL term
        residual = (target - gamma).pow(2)
        nll = (
            0.5 * torch.log(math.pi / nu)
            - alpha * torch.log(Omega)
            + (alpha + 0.5) * torch.log(residual * nu + Omega)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )

        # Regularization: penalize evidence on wrong predictions
        reg = (target - gamma).abs() * (2.0 * nu + alpha)

        loss = nll + self.coeff * reg
        return loss.mean()


# ═══════════════════════════════════════════════════════════════
#  L_DAG: DAG Root Formula Cross-Entropy (P3-T5)
# ═══════════════════════════════════════════════════════════════

class DAGRootCE(nn.Module):
    """
    Cross-entropy loss on the DAG-GNN root embedding prediction.

    The DAG-GNN encodes fragmentation trees bottom-up and produces
    a root embedding h_dag. This loss trains a linear classifier on
    h_dag to predict the correct molecular formula class.

    Only active when has_fragments=True for the sample.
    """

    def __init__(self, h_dag_dim: int = 128, num_formulas: int = 1000):
        super().__init__()
        self.classifier = nn.Linear(h_dag_dim, num_formulas)

    def forward(
        self,
        h_dag: torch.Tensor,           # [B, h_dag_dim]
        formula_labels: torch.Tensor,   # [B] long — formula index
        mask: Optional[torch.Tensor] = None,  # [B] bool — has_fragments
    ) -> torch.Tensor:
        if mask is not None:
            h_dag = h_dag[mask]
            formula_labels = formula_labels[mask]

        if h_dag.numel() == 0:
            return torch.tensor(0.0, device=h_dag.device, requires_grad=True)

        logits = self.classifier(h_dag)
        return F.cross_entropy(logits, formula_labels)


# ═══════════════════════════════════════════════════════════════
#  L_grammar: Mean Grammar Violation Penalty (P3-T5)
# ═══════════════════════════════════════════════════════════════

class GrammarViolationLoss(nn.Module):
    """
    Penalizes candidates whose formula is incompatible with observed
    neutral losses in the fragmentation DAG.

    Uses compute_grammar_violation_score() from dag_gnn module.
    The loss is mean violation score across all candidates, weighted
    by their EBM probability so high-probability wrong assignments
    are penalized more.

    Only active when has_fragments=True for the sample.
    """

    def forward(
        self,
        grammar_scores: torch.Tensor,  # [N] violation scores (0=perfect, 1=bad)
        log_probs: torch.Tensor,       # [N] log P(F) from GIN
        mask: Optional[torch.Tensor] = None,  # [N] bool — has_fragments
    ) -> torch.Tensor:
        if mask is not None:
            grammar_scores = grammar_scores[mask]
            log_probs = log_probs[mask]

        if grammar_scores.numel() == 0:
            return torch.tensor(0.0, device=grammar_scores.device, requires_grad=True)

        # Probability-weighted mean grammar violation
        probs = log_probs.detach().exp()
        weighted = (grammar_scores * probs).sum() / probs.sum().clamp(min=1e-8)
        return weighted


# ═══════════════════════════════════════════════════════════════
#  L_physics: PINN Mass Conservation MSE (P3-T5)
# ═══════════════════════════════════════════════════════════════

class PhysicsMassConservation(nn.Module):
    """
    Physics-Informed Neural Network (PINN) loss for mass conservation.

    Enforces that predicted neutral loss masses sum up correctly:
      m(precursor) = m(fragment_i) + m(neutral_loss_i)

    L_physics = MSE(Σ fragment + neutral_loss, precursor_mass)

    Only active when has_fragments=True for the sample.
    """

    def forward(
        self,
        precursor_mass: torch.Tensor,   # [B] precursor m/z
        fragment_masses: torch.Tensor,   # [B, max_frags] fragment m/z
        neutral_losses: torch.Tensor,    # [B, max_frags] predicted NL masses
        frag_mask: Optional[torch.Tensor] = None,  # [B, max_frags] bool — valid fragments
    ) -> torch.Tensor:
        if fragment_masses.numel() == 0:
            return torch.tensor(0.0, device=precursor_mass.device, requires_grad=True)

        # Expected: fragment + neutral_loss = precursor for each row
        reconstructed = fragment_masses + neutral_losses  # [B, max_frags]

        if frag_mask is not None:
            # Only compute loss on valid fragment entries
            diff = (reconstructed - precursor_mass.unsqueeze(-1)) ** 2
            diff = diff * frag_mask.float()
            n_valid = frag_mask.float().sum().clamp(min=1.0)
            return diff.sum() / n_valid
        else:
            return F.mse_loss(
                reconstructed,
                precursor_mass.unsqueeze(-1).expand_as(reconstructed),
            )


# ═══════════════════════════════════════════════════════════════
#  L_CCS: CCS Prediction MSE (P3-T6)
# ═══════════════════════════════════════════════════════════════

class CCSPredictionLoss(nn.Module):
    """
    MSE between predicted and observed CCS (Å²).

    Only active when has_ims=True for the sample.
    Imported from ccs_scorer.CCSLoss but kept here for registry.
    """

    def forward(
        self,
        ccs_pred: torch.Tensor,      # [N] predicted CCS
        ccs_observed: torch.Tensor,   # [N] observed CCS
        mask: Optional[torch.Tensor] = None,  # [N] bool — has_ims
    ) -> torch.Tensor:
        if mask is not None:
            ccs_pred = ccs_pred[mask]
            ccs_observed = ccs_observed[mask]

        if ccs_pred.numel() == 0:
            return torch.tensor(0.0, device=ccs_pred.device, requires_grad=True)

        return F.mse_loss(ccs_pred, ccs_observed)


# ═══════════════════════════════════════════════════════════════
#  L_margin: Pairwise Rank-Margin Hinge Loss (Run11)
# ═══════════════════════════════════════════════════════════════

class RankMarginLoss(nn.Module):
    """
    Pairwise rank-margin hinge loss.

    For each spectral feature, enforces that the correct formula's energy
    is separated from the lowest-energy wrong formula by at least `margin`:

        L_margin = mean_features max(0, E_correct - E_hardest_wrong + margin)

    Motivation: 65% of CHONS errors are rank=2 (near-miss). The model's energy
    for the correct formula is only marginally higher than the best wrong formula.
    This loss directly penalises that gap and is weighted more heavily for
    element classes with documented near-miss failure (CHONS, CHONSP).

    Args:
        margin:         Minimum required energy gap (default 0.5).
        focus_classes:  Set of element-class indices that receive double weight.
                        Default {3, 5} = CHONS (3), CHONSP (5).
        max_loss:       Per-feature loss cap to prevent domination.
    """

    def __init__(
        self,
        margin: float = 0.5,
        focus_classes: Optional[set] = None,
        max_loss: float = 10.0,
    ):
        super().__init__()
        self.margin = margin
        # CHONS=3, CHONSP=5 in ELEMENTAL_CLASSES ordering
        self.focus_classes: set = focus_classes if focus_classes is not None else {3, 5}
        self.max_loss = max_loss

    def forward(
        self,
        energies: torch.Tensor,              # [total_candidates]
        candidate_to_feature: torch.Tensor,  # [total_candidates] long
        correct_mask: torch.Tensor,          # [total_candidates] bool
        eclass_labels: Optional[torch.Tensor] = None,  # [total_candidates] long
    ) -> torch.Tensor:
        """
        Compute mean pairwise hinge loss over all features.

        For features in focus_classes, the loss is weighted 2× to emphasise
        the CHONS and CHONSP failure modes.
        """
        device = energies.device
        correct_indices = correct_mask.nonzero(as_tuple=True)[0]
        if correct_indices.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        correct_features = candidate_to_feature[correct_indices]
        unique_features = correct_features.unique()

        losses = []
        for feat_idx in unique_features:
            feat_mask = candidate_to_feature == feat_idx
            correct_feat = correct_mask & feat_mask
            wrong_feat = (~correct_mask) & feat_mask

            if not correct_feat.any() or not wrong_feat.any():
                continue

            # Lowest energy correct formula (most confidently correct)
            e_correct = energies[correct_feat].min()
            # Lowest energy wrong formula (hardest negative)
            e_wrong_min = energies[wrong_feat].min()

            hinge = F.relu(e_correct - e_wrong_min + self.margin)

            # Double weight for near-miss element classes
            if eclass_labels is not None:
                feat_eclass = eclass_labels[correct_feat][0].item()
                if int(feat_eclass) in self.focus_classes:
                    hinge = hinge * 2.0

            losses.append(hinge.clamp(max=self.max_loss))

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()



class MOSAICLoss(nn.Module):
    """
    Full multi-term loss for MOSAIC-GNN.

    L_total = λ_ce1·L_CE1 + λ_ce2·L_CE2 + λ_ebm·L_EBM
              + λ_adc·L_ADC + λ_iso·L_ISO + λ_uq·L_UQ + λ_dom·L_DOM
              + λ_dag·L_DAG + λ_gram·L_grammar + λ_phys·L_physics + λ_ccs·L_CCS

    Supports:
      - Automatic λ_ADC annealing: init → final linearly over first N epochs
      - Per-term logging
      - Configurable loss weights for rebalancing ranking vs classification
      - Option to use corrected_energies for L_EBM (matching inference ranking)
      - Conditional activation: L_DAG/L_grammar/L_physics only when has_fragments
      - Conditional activation: L_CCS only when has_ims
    """

    def __init__(
        self,
        lambda_adc: float = 0.1,
        lambda_adc_final: float = 0.5,
        lambda_adc_anneal_epoch: int = 20,
        lambda_iso: float = 0.2,
        lambda_uq: float = 0.05,
        lambda_dom: float = 0.1,
        lambda_ebm: float = 1.0,
        lambda_ce1: float = 1.0,
        lambda_ce2: float = 1.0,
        lambda_dag: float = 0.15,
        lambda_grammar: float = 0.05,
        lambda_physics: float = 1.0,
        lambda_ccs: float = 0.1,
        num_hard_negatives: int = 10,
        num_random_negatives: int = 10,
        infonce_temperature: float = 1.0,
        use_corrected_energies_for_ebm: bool = False,
        num_elemental_classes: int = 7,
        num_dbe_bins: int = 4,
        evidential_coeff: float = 0.05,
        min_alpha: float = 1.01,
        label_smoothing: float = 0.0,
        h_dag_dim: int = 128,
        num_dag_formulas: int = 1000,
        lambda_margin: float = 0.0,
        margin_value: float = 0.5,
    ):
        super().__init__()

        # Sub-losses
        self.ce1 = ElementalClassCE(num_classes=num_elemental_classes,
                                    label_smoothing=label_smoothing)
        self.ce2 = DBEBinCE(num_bins=num_dbe_bins,
                            label_smoothing=label_smoothing)
        self.infonce = InfoNCELoss(
            num_hard_negatives=num_hard_negatives,
            num_random_negatives=num_random_negatives,
            temperature=infonce_temperature,
            max_loss=50.0,
        )
        # Temperature for CE logits — with LayerNorm in HierarchicalHead,
        # logits are already well-scaled, so temperature=1.0 is sufficient.
        self.ce_temperature = 1.0
        self.iso_loss = IsotopePhysicsLoss()
        self.uq_loss = EvidentialNIGLoss(
            coeff=evidential_coeff,
            min_alpha=min_alpha,
        )

        # P3 loss terms
        self.dag_root_ce = DAGRootCE(
            h_dag_dim=h_dag_dim,
            num_formulas=num_dag_formulas,
        )
        self.grammar_loss = GrammarViolationLoss()
        self.physics_loss = PhysicsMassConservation()
        self.ccs_loss = CCSPredictionLoss()
        self.rank_margin_loss = RankMarginLoss(margin=margin_value)

        # Loss weights
        self.lambda_adc_init = lambda_adc
        self.lambda_adc_final = lambda_adc_final
        self.lambda_adc_anneal_epoch = lambda_adc_anneal_epoch
        self.lambda_iso = lambda_iso
        self.lambda_uq = lambda_uq
        self.lambda_dom = lambda_dom
        self.lambda_ebm = lambda_ebm
        self.lambda_ce1 = lambda_ce1
        self.lambda_ce2 = lambda_ce2
        self.lambda_dag = lambda_dag
        self.lambda_grammar = lambda_grammar
        self.lambda_physics = lambda_physics
        self.lambda_ccs = lambda_ccs
        self.lambda_margin = lambda_margin
        self.use_corrected_energies_for_ebm = use_corrected_energies_for_ebm

        self._current_epoch = 0

    @property
    def lambda_adc(self) -> float:
        """Linearly anneal λ_ADC from init to final over the annealing period."""
        if self._current_epoch >= self.lambda_adc_anneal_epoch:
            return self.lambda_adc_final
        t = self._current_epoch / max(self.lambda_adc_anneal_epoch, 1)
        return self.lambda_adc_init + t * (self.lambda_adc_final - self.lambda_adc_init)

    def set_epoch(self, epoch: int):
        """Update current epoch for annealing schedules."""
        self._current_epoch = epoch

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute full multi-term loss.

        Args:
            model_outputs: dict from MOSAICGNN.forward():
                - l1_logits:   [N, 7]   elemental class logits
                - l2_logits:   [N, 4]   DBE bin logits
                - energies:    [total]   raw energies
                - L_adc:       scalar    adduct consistency loss (from Module 4)
                - theo_m1:     [N_iso]  predicted M+1
                - theo_m2:     [N_iso]  predicted M+2

            targets: dict with ground-truth labels:
                - eclass_labels:        [N] long  elemental class GT
                - dbe_bin_labels:       [N] long  DBE bin GT
                - correct_mask:         [total] bool  correct formula mask
                - candidate_to_feature: [total] long  feature assignment
                - obs_m1:               [N_iso] observed M+1 ratio
                - obs_m2:               [N_iso] observed M+2 ratio
                - iso_mask:             [N_iso] bool  valid ISO entries

                # Evidential outputs (optional, Phase 3)
                - uq_gamma:  [N] predicted mean
                - uq_nu:     [N] evidence count
                - uq_alpha:  [N] shape
                - uq_beta:   [N] rate
                - uq_target: [N] target

        Returns:
            (L_total, loss_dict) where loss_dict logs each term.
        """
        device = model_outputs['energies'].device
        loss_dict = {}

        # ── L_CE1: Elemental class cross-entropy ──
        if 'l1_logits' in model_outputs and 'eclass_labels' in targets:
            L_CE1 = self.ce1(
                model_outputs['l1_logits'] / self.ce_temperature,
                targets['eclass_labels'],
            )
        else:
            L_CE1 = torch.tensor(0.0, device=device)
        loss_dict['L_CE1'] = L_CE1.item()

        # ── L_CE2: DBE bin cross-entropy ──
        if 'l2_logits' in model_outputs and 'dbe_bin_labels' in targets:
            L_CE2 = self.ce2(
                model_outputs['l2_logits'] / self.ce_temperature,
                targets['dbe_bin_labels'],
            )
        else:
            L_CE2 = torch.tensor(0.0, device=device)
        loss_dict['L_CE2'] = L_CE2.item()

        # ── L_EBM: InfoNCE contrastive loss ──
        if 'energies' in model_outputs and 'correct_mask' in targets:
            # Choose which energies to train on:
            # corrected_energies matches inference ranking pipeline
            if self.use_corrected_energies_for_ebm and 'corrected_energies' in model_outputs:
                ebm_energies = model_outputs['corrected_energies']
            else:
                ebm_energies = model_outputs['energies']
            # Get FCL lattice edges for hard negative mining
            edge_index = model_outputs.get('fcl_edge_index', None)
            if edge_index is None:
                edge_index = targets.get('fcl_edge_index', None)
            L_EBM = self.infonce(
                ebm_energies,
                targets['candidate_to_feature'],
                targets['correct_mask'],
                edge_index=edge_index,
            )
        else:
            L_EBM = torch.tensor(0.0, device=device)
        loss_dict['L_EBM'] = L_EBM.item()

        # ── L_ADC: Adduct consistency (already computed in Module 4) ──
        L_ADC = model_outputs.get('L_adc', torch.tensor(0.0, device=device))
        L_ADC = torch.clamp(L_ADC, max=10.0)   # raw ~2000; cap like ISO/InfoNCE
        loss_dict['L_ADC'] = L_ADC.item()

        # ── L_ISO: Isotope physics MSE ──
        if 'theo_m1' in model_outputs and 'obs_m1' in targets:
            L_ISO = self.iso_loss(
                model_outputs['theo_m1'],
                model_outputs['theo_m2'],
                targets['obs_m1'],
                targets['obs_m2'],
                mask=targets.get('iso_mask', None),
            )
        else:
            L_ISO = torch.tensor(0.0, device=device)
        loss_dict['L_ISO'] = L_ISO.item()

        # ── L_UQ: Evidential NIG loss (Phase 3, optional) ──
        # The evidential head produces (gamma, nu, alpha, beta) in model_outputs.
        # The target is the correct-formula energy (or 0 for correct, 1 for wrong).
        if all(k in model_outputs for k in ['uq_gamma', 'uq_nu', 'uq_alpha', 'uq_beta']):
            # Build regression target: 0 for correct candidates, 1 for wrong
            if 'correct_mask' in targets:
                uq_target = (~targets['correct_mask']).float()
            elif 'uq_target' in targets:
                uq_target = targets['uq_target']
            else:
                uq_target = torch.zeros_like(model_outputs['uq_gamma'])
            L_UQ = self.uq_loss(
                model_outputs['uq_gamma'],
                model_outputs['uq_nu'],
                model_outputs['uq_alpha'],
                model_outputs['uq_beta'],
                uq_target,
            )
        elif all(k in targets for k in ['uq_gamma', 'uq_nu', 'uq_alpha', 'uq_beta', 'uq_target']):
            # Fallback: legacy path where UQ params are in targets
            L_UQ = self.uq_loss(
                targets['uq_gamma'],
                targets['uq_nu'],
                targets['uq_alpha'],
                targets['uq_beta'],
                targets['uq_target'],
            )
        else:
            L_UQ = torch.tensor(0.0, device=device)
        loss_dict['L_UQ'] = L_UQ.item()

        # ── L_DOM: Domain adversarial loss (Phase 3, optional) ──
        L_DOM = torch.tensor(0.0, device=device)
        if 'domain_logits' in model_outputs and 'domain_target' in model_outputs:
            L_DOM = F.cross_entropy(
                model_outputs['domain_logits'],
                model_outputs['domain_target'],
            )
        loss_dict['L_DOM'] = L_DOM.item()

        # ── L_DAG: DAG root formula CE (P3-T5, optional) ──
        L_DAG = torch.tensor(0.0, device=device)
        if 'h_dag' in model_outputs and 'dag_formula_labels' in targets:
            frag_mask = targets.get('has_fragments_mask', None)
            L_DAG = self.dag_root_ce(
                model_outputs['h_dag'],
                targets['dag_formula_labels'],
                mask=frag_mask,
            )
        loss_dict['L_DAG'] = L_DAG.item()

        # ── L_grammar: Grammar violation penalty (P3-T5, optional) ──
        L_grammar = torch.tensor(0.0, device=device)
        if 'grammar_scores' in model_outputs and 'log_probs' in model_outputs:
            frag_mask = targets.get('has_fragments_mask', None)
            L_grammar = self.grammar_loss(
                model_outputs['grammar_scores'],
                model_outputs['log_probs'],
                mask=frag_mask,
            )
        loss_dict['L_grammar'] = L_grammar.item()

        # ── L_physics: Mass conservation (P3-T5, optional) ──
        L_physics = torch.tensor(0.0, device=device)
        if all(k in model_outputs for k in ['precursor_mass', 'fragment_masses', 'neutral_losses']):
            L_physics = self.physics_loss(
                model_outputs['precursor_mass'],
                model_outputs['fragment_masses'],
                model_outputs['neutral_losses'],
                frag_mask=model_outputs.get('frag_mask', None),
            )
        loss_dict['L_physics'] = L_physics.item()

        # ── L_CCS: CCS prediction MSE (P3-T6, optional) ──
        L_CCS = torch.tensor(0.0, device=device)
        if 'ccs_pred' in model_outputs and 'ccs_observed' in targets:
            ims_mask = targets.get('has_ims_mask', None)
            L_CCS = self.ccs_loss(
                model_outputs['ccs_pred'],
                targets['ccs_observed'],
                mask=ims_mask,
            )
        loss_dict['L_CCS'] = L_CCS.item()

        # ── L_margin: Rank-margin hinge (Run11, optional) ──
        L_margin = torch.tensor(0.0, device=device)
        if self.lambda_margin > 0 and 'correct_mask' in targets:
            ebm_e = (
                model_outputs['corrected_energies']
                if self.use_corrected_energies_for_ebm and 'corrected_energies' in model_outputs
                else model_outputs['energies']
            )
            L_margin = self.rank_margin_loss(
                ebm_e,
                targets['candidate_to_feature'],
                targets['correct_mask'],
                eclass_labels=targets.get('eclass_labels', None),
            )
        loss_dict['L_margin'] = L_margin.item()

        # ── Total loss with annealing ──
        lam_adc = self.lambda_adc
        loss_dict['lambda_adc_current'] = lam_adc

        L_total = (
            self.lambda_ce1 * L_CE1
            + self.lambda_ce2 * L_CE2
            + self.lambda_ebm * L_EBM
            + lam_adc * L_ADC
            + self.lambda_iso * L_ISO
            + self.lambda_uq * L_UQ
            + self.lambda_dom * L_DOM
            + self.lambda_dag * L_DAG
            + self.lambda_grammar * L_grammar
            + self.lambda_physics * L_physics
            + self.lambda_ccs * L_CCS
            + self.lambda_margin * L_margin
        )
        loss_dict['L_total'] = L_total.item()

        return L_total, loss_dict

    def get_loss_summary(self, loss_dict: Dict[str, float]) -> str:
        """Format loss breakdown for logging."""
        parts = [
            f"L_total={loss_dict['L_total']:.4f}",
            f"CE1={loss_dict['L_CE1']:.4f}",
            f"CE2={loss_dict['L_CE2']:.4f}",
            f"EBM={loss_dict['L_EBM']:.4f}",
            f"ADC={loss_dict['L_ADC']:.4f}(λ={loss_dict['lambda_adc_current']:.3f})",
            f"ISO={loss_dict['L_ISO']:.4f}",
            f"UQ={loss_dict['L_UQ']:.4f}",
            f"DOM={loss_dict.get('L_DOM', 0):.4f}",
            f"DAG={loss_dict.get('L_DAG', 0):.4f}",
            f"GRAM={loss_dict.get('L_grammar', 0):.4f}",
            f"PHYS={loss_dict.get('L_physics', 0):.4f}",
            f"CCS={loss_dict.get('L_CCS', 0):.4f}",
            f"MARGIN={loss_dict.get('L_margin', 0):.4f}",
        ]
        return " | ".join(parts)
