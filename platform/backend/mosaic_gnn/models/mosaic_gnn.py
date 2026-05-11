"""
MOSAIC-GNN: Full Model Assembly
================================
Combines all five core modules plus Phase-3 extensions:

  Module 1: SMDG R-GCN Encoder    → h_mix  (mixture context)
  Module 2: Isotope GAT Encoder    → p_iso  (heteroatom probs)
  Module 3: Formula GIN Scorer     → energies, log_probs
  Module 4: Adduct Consistency     → corrected energies
  Module 5: Hierarchical Head      → L1/L2 logits + masked ranking

Phase-3 extensions (opt-in):
  P3-T3: Evidential UQ Head        → (γ, ν, α, β), uncertainties
  P3-T2: FiLM Domain Conditioning  → matrix-adaptive embeddings
  P3-T2: Domain Adversarial Head   → gradient-reversal classifier
  P3-T5: DAG-GNN Fragmentation     → h_dag encoding
  P3-T6: IMS/CCS Joint Scoring     → CCS-aware energy scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data, Batch
from typing import Dict, Optional, Tuple, List

from .smdg_rgcn import SMDGEncoder
from .iso_gat import IsotopeGATEncoder
from .fcl_gin import FormulaGINScorer
from .adc_layer import AdductConsistencyLayer
from .hierarchical_head import HierarchicalHead
from .evidential_head import EvidentialHead
from .film_conditioning import (
    FiLMLayer, MatrixEmbedding, DomainClassifier, NUM_MATRIX_TYPES,
)
from .dag_gnn import DAGGNNEncoder, make_empty_dag
from .ccs_scorer import JointScorer


class MOSAICGNN(nn.Module):
    """
    MOSAIC-GNN: Mixture-Optimized Spectral-Adduct-Isotope Co-inference GNN.

    End-to-end model for MS1-only molecular formula assignment.

    Phase 3 extensions (all default-off for backward compat):
        use_evidential:         Evidential NIG uncertainty head
        use_film:               FiLM matrix-type conditioning on SMDG
        use_domain_adversarial: Gradient-reversal domain classifier
    """

    def __init__(
        self,
        # Module 1: SMDG R-GCN
        smdg_in_channels: int = 4,
        smdg_hidden_dim: int = 256,
        smdg_num_layers: int = 4,
        smdg_num_bases: int = 4,
        smdg_num_relations: int = 11,
        smdg_dropout: float = 0.1,
        # Module 2: Isotope GAT
        iso_in_channels: int = 4,
        iso_hidden_dim: int = 128,
        iso_num_layers: int = 3,
        iso_num_heads: int = 8,
        iso_dropout: float = 0.1,
        iso_output_dim: int = 9,
        # Module 3: GIN Scorer
        gin_formula_feat_dim: int = 14,
        gin_hidden_dim: int = 256,
        gin_num_layers: int = 3,
        gin_eps_init: float = 0.0,
        gin_dropout: float = 0.1,
        gin_use_bypass: bool = True,
        # Module 4: ADC
        adc_lambda_init: float = 0.1,
        # Module 5: Hierarchical Head
        hier_num_eclass: int = 7,
        hier_num_dbe_bins: int = 4,
        hier_dropout: float = 0.1,
        # ── Phase 3 ──
        use_evidential: bool = False,
        use_film: bool = False,
        use_domain_adversarial: bool = False,
        film_embed_dim: int = 64,
        num_matrix_types: int = NUM_MATRIX_TYPES,
        # P3-T5: DAG-GNN
        use_dag_gnn: bool = False,
        dag_node_dim: int = 5,
        dag_edge_dim: int = 16,
        dag_hidden_dim: int = 64,
        dag_output_dim: int = 128,
        # P3-T6: CCS Joint Scoring
        use_ccs: bool = False,
        ccs_input_dim: int = 13,
        ccs_hidden_dim: int = 128,
        ccs_alpha_init: float = 0.5,
    ):
        super().__init__()
        self.use_evidential = use_evidential
        self.use_film = use_film
        self.use_domain_adversarial = use_domain_adversarial
        self.use_dag_gnn = use_dag_gnn
        self.use_ccs = use_ccs

        # Module 1: SMDG R-GCN Encoder
        self.smdg_encoder = SMDGEncoder(
            in_channels=smdg_in_channels,
            hidden_dim=smdg_hidden_dim,
            num_layers=smdg_num_layers,
            num_bases=smdg_num_bases,
            num_relations=smdg_num_relations,
            dropout=smdg_dropout,
        )

        # Module 2: Isotope GAT Encoder
        self.iso_encoder = IsotopeGATEncoder(
            in_channels=iso_in_channels,
            hidden_dim=iso_hidden_dim,
            num_layers=iso_num_layers,
            num_heads=iso_num_heads,
            dropout=iso_dropout,
            output_dim=iso_output_dim,
        )

        # Module 3: Formula GIN Scorer
        gin_h_dag_dim = dag_output_dim if use_dag_gnn else 0
        self.gin_scorer = FormulaGINScorer(
            formula_feat_dim=gin_formula_feat_dim,
            h_mix_dim=smdg_hidden_dim,
            p_iso_dim=iso_output_dim,
            h_dag_dim=gin_h_dag_dim,
            hidden_dim=gin_hidden_dim,
            num_layers=gin_num_layers,
            eps_init=gin_eps_init,
            dropout=gin_dropout,
            use_bypass=gin_use_bypass,
        )

        # Module 4: Adduct Consistency Layer
        self.adc_layer = AdductConsistencyLayer(
            lambda_init=adc_lambda_init,
        )

        # Module 5: Hierarchical Classification Head
        self.hier_head = HierarchicalHead(
            hidden_dim=gin_hidden_dim,
            num_elemental_classes=hier_num_eclass,
            num_dbe_bins=hier_num_dbe_bins,
            dropout=hier_dropout,
        )

        # ── Phase 3: Evidential UQ Head ──
        if use_evidential:
            self.evidential_head = EvidentialHead(
                hidden_dim=gin_hidden_dim,
                min_alpha=1.01,
                dropout=gin_dropout,
            )

        # ── Phase 3: FiLM Conditioning ──
        if use_film:
            self.matrix_embed = MatrixEmbedding(
                num_types=num_matrix_types,
                embed_dim=film_embed_dim,
            )
            # One FiLM layer applied after the SMDG encoder
            self.film_layer = FiLMLayer(
                cond_dim=film_embed_dim,
                hidden_dim=smdg_hidden_dim,
            )

        # ── Phase 3: Domain Adversarial Classifier ──
        if use_domain_adversarial:
            self.domain_classifier = DomainClassifier(
                hidden_dim=smdg_hidden_dim,
                num_domains=num_matrix_types,
                dropout=smdg_dropout,
            )

        # ── P3-T5: DAG-GNN Fragmentation Encoder ──
        if use_dag_gnn:
            self.dag_encoder = DAGGNNEncoder(
                node_feat_dim=dag_node_dim,
                edge_feat_dim=dag_edge_dim,
                hidden_dim=dag_hidden_dim,
                out_dim=dag_output_dim,
            )

        # ── P3-T6: CCS Joint Scorer ──
        if use_ccs:
            self.joint_scorer = JointScorer(
                ccs_input_dim=ccs_input_dim,
                ccs_hidden_dim=ccs_hidden_dim,
                alpha_init=ccs_alpha_init,
            )

    def forward(
        self,
        smdg_data: HeteroData,
        iso_batch: Batch,
        fcl_batch: Batch,
        adc_edge_index: Optional[torch.Tensor] = None,
        feature_mz: Optional[torch.Tensor] = None,
        candidate_masses: Optional[torch.Tensor] = None,
        candidate_to_feature: Optional[torch.Tensor] = None,
        candidate_eclass: Optional[torch.Tensor] = None,
        candidate_dbe_bin: Optional[torch.Tensor] = None,
        iso_feature_mapping: Optional[torch.Tensor] = None,
        matrix_type_idx: Optional[torch.Tensor] = None,
        grl_alpha: float = 1.0,
        # P3-T5/T6 extensions
        has_fragments: bool = False,
        has_ims: bool = False,
        dag_data: Optional[Data] = None,
        fragment_mz: Optional[torch.Tensor] = None,
        precursor_mz: Optional[float] = None,
        ccs_observed: Optional[torch.Tensor] = None,
        ccs_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all modules (P2 core + P3 extensions).

        New P3 args:
            matrix_type_idx: scalar LongTensor — matrix type for FiLM.
            grl_alpha: float — GRL scaling for domain adversarial.
            has_fragments: bool — whether this sample has MS2/fragment data.
            has_ims: bool — whether this sample has IMS/CCS data.
            dag_data: Data — fragmentation DAG graph (P3-T5).
            fragment_mz: Tensor — fragment m/z values (P3-T5).
            precursor_mz: float — precursor m/z (P3-T5).
            ccs_observed: Tensor — observed CCS values (P3-T6).
            ccs_features: Tensor — CCS predictor input features (P3-T6).
        """
        outputs = {}

        # ── Module 1: SMDG R-GCN ──
        h_mix = self.smdg_encoder(smdg_data)  # [N_features, 256]

        # ── P3: FiLM conditioning ──
        if self.use_film and matrix_type_idx is not None:
            z_matrix = self.matrix_embed(matrix_type_idx)  # [embed_dim]
            h_mix = self.film_layer(h_mix, z_matrix)

        outputs['h_mix'] = h_mix

        # ── P3: Domain adversarial classifier ──
        if self.use_domain_adversarial and matrix_type_idx is not None:
            domain_logits = self.domain_classifier(h_mix, alpha=grl_alpha)
            # Mean-pool per-node logits to sample-level
            outputs['domain_logits'] = domain_logits.mean(dim=0, keepdim=True)
            outputs['domain_target'] = matrix_type_idx.unsqueeze(0) \
                if matrix_type_idx.dim() == 0 else matrix_type_idx

        # ── Module 2: Isotope GAT ──
        p_iso, theo_m1, theo_m2 = self.iso_encoder(iso_batch)
        outputs['p_iso'] = p_iso
        outputs['theo_m1'] = theo_m1
        outputs['theo_m2'] = theo_m2

        # Map ISO outputs back to feature space
        N_features = h_mix.size(0)
        if iso_feature_mapping is not None and iso_feature_mapping.numel() > 0:
            p_iso_full = torch.zeros(N_features, p_iso.size(1), device=h_mix.device)
            p_iso_full[iso_feature_mapping] = p_iso
        else:
            p_iso_full = p_iso[:N_features] if p_iso.size(0) >= N_features \
                else F.pad(p_iso, (0, 0, 0, N_features - p_iso.size(0)))
        outputs['p_iso_full'] = p_iso_full

        # ── P3-T5: DAG-GNN Fragmentation Encoding ──
        h_dag = None
        if self.use_dag_gnn and has_fragments and dag_data is not None:
            h_dag = self.dag_encoder(dag_data)  # [N_features, dag_output_dim]
            outputs['h_dag'] = h_dag

        # ── Module 3: GIN Scorer ──
        energies, log_probs, h_cand = self.gin_scorer(
            fcl_batch, h_mix=h_mix, p_iso=p_iso_full, h_dag=h_dag,
            return_hidden=True,
        )
        outputs['energies'] = energies
        outputs['log_probs'] = log_probs

        # ── P3-T6: CCS Joint Scoring ──
        if self.use_ccs and ccs_features is not None:
            e_joint, ccs_pred = self.joint_scorer(
                energies, ccs_features,
                ccs_observed=ccs_observed,
                has_ims=has_ims,
            )
            outputs['ccs_pred'] = ccs_pred
            outputs['energies_with_ccs'] = e_joint
        else:
            outputs['energies_with_ccs'] = energies

        # ── P3: Evidential UQ ──
        if self.use_evidential:
            evid = self.evidential_head(h_cand)
            outputs['uq_gamma'] = evid['gamma']
            outputs['uq_nu'] = evid['nu']
            outputs['uq_alpha'] = evid['alpha']
            outputs['uq_beta'] = evid['beta']
            outputs['u_ale'] = evid['u_ale']
            outputs['u_epi'] = evid['u_epi']

        # Apply hard physics constraints
        constrained_energies = self.gin_scorer.apply_hard_constraints(energies, fcl_batch)
        outputs['constrained_energies'] = constrained_energies

        # ── Module 4: ADC Consistency ──
        if adc_edge_index is not None and adc_edge_index.numel() > 0 \
                and candidate_masses is not None and feature_mz is not None:
            corrected_energies, L_adc = self.adc_layer(
                constrained_energies, candidate_masses, feature_mz,
                adc_edge_index, candidate_to_feature=candidate_to_feature,
            )
        else:
            corrected_energies = constrained_energies
            L_adc = torch.tensor(0.0, device=energies.device)
        outputs['corrected_energies'] = corrected_energies
        outputs['L_adc'] = L_adc

        # ── Module 5: Hierarchical Head ──
        l1_logits, l2_logits = self.hier_head(
            h_cand, candidate_eclass=candidate_eclass, candidate_dbe_bin=candidate_dbe_bin,
        )
        outputs['l1_logits'] = l1_logits
        outputs['l2_logits'] = l2_logits

        return outputs

    @torch.no_grad()
    def predict(
        self,
        smdg_data: HeteroData,
        iso_batch: Batch,
        fcl_batch: Batch,
        adc_edge_index: Optional[torch.Tensor] = None,
        feature_mz: Optional[torch.Tensor] = None,
        candidate_masses: Optional[torch.Tensor] = None,
        candidate_to_feature: Optional[torch.Tensor] = None,
        candidate_eclass_labels: Optional[torch.Tensor] = None,
        candidate_dbe_bins: Optional[torch.Tensor] = None,
        matrix_type_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: predict formula with hierarchical masking + UQ.
        """
        self.eval()
        outputs = self.forward(
            smdg_data, iso_batch, fcl_batch,
            adc_edge_index=adc_edge_index,
            feature_mz=feature_mz,
            candidate_masses=candidate_masses,
            candidate_to_feature=candidate_to_feature,
            matrix_type_idx=matrix_type_idx,
        )

        # Apply hierarchical mask
        if candidate_eclass_labels is not None and candidate_dbe_bins is not None:
            masked = self.hier_head.apply_hierarchical_mask(
                outputs['corrected_energies'],
                outputs['l1_logits'],
                outputs['l2_logits'],
                candidate_eclass_labels,
                candidate_dbe_bins,
                fcl_batch.batch,
            )
            outputs['masked_energies'] = masked

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Report parameter breakdown by module."""
        counts = {}
        for name, module in [
            ('Module1_SMDG_RGCN', self.smdg_encoder),
            ('Module2_ISO_GAT', self.iso_encoder),
            ('Module3_FCL_GIN', self.gin_scorer),
            ('Module4_ADC', self.adc_layer),
            ('Module5_Hierarchical', self.hier_head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        # P3 modules
        if self.use_evidential:
            counts['P3_Evidential'] = sum(
                p.numel() for p in self.evidential_head.parameters()
            )
        if self.use_film:
            counts['P3_FiLM'] = sum(
                p.numel() for p in self.matrix_embed.parameters()
            ) + sum(
                p.numel() for p in self.film_layer.parameters()
            )
        if self.use_domain_adversarial:
            counts['P3_DomainAdv'] = sum(
                p.numel() for p in self.domain_classifier.parameters()
            )
        if self.use_dag_gnn:
            counts['P3_DAG_GNN'] = sum(
                p.numel() for p in self.dag_encoder.parameters()
            )
        if self.use_ccs:
            counts['P3_CCS_Scorer'] = sum(
                p.numel() for p in self.joint_scorer.parameters()
            )
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
