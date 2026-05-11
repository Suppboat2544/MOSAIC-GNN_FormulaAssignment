"""
MOSAIC-GNN: Model components
==============================
All GNN modules for the MOSAIC-GNN pipeline (P2 core + P3 extensions).
"""

from .smdg_rgcn import BasisRGCNConv, SMDGEncoder
from .iso_gat import DifferentiableRoepstorff, IsotopeGATEncoder
from .fcl_gin import GINLayer, FormulaGINScorer
from .adc_layer import AdductConsistencyLayer
from .hierarchical_head import HierarchicalHead
from .mosaic_gnn import MOSAICGNN
from .evidential_head import EvidentialHead
from .film_conditioning import (
    FiLMLayer, MatrixEmbedding, DomainClassifier,
    gradient_reversal, infer_matrix_type,
    MATRIX_TYPES, MATRIX_TO_IDX, NUM_MATRIX_TYPES,
)
from .dag_gnn import DAGGNNEncoder, build_fragmentation_dag, compute_grammar_violation_score, make_empty_dag
from .ccs_scorer import SchNetCCSPredictor, JointScorer, CCSLoss, build_ccs_features

__all__ = [
    'BasisRGCNConv',
    'SMDGEncoder',
    'DifferentiableRoepstorff',
    'IsotopeGATEncoder',
    'GINLayer',
    'FormulaGINScorer',
    'AdductConsistencyLayer',
    'HierarchicalHead',
    'MOSAICGNN',
    # Phase 3
    'EvidentialHead',
    'FiLMLayer',
    'MatrixEmbedding',
    'DomainClassifier',
    'gradient_reversal',
    'infer_matrix_type',
    'MATRIX_TYPES',
    'MATRIX_TO_IDX',
    'NUM_MATRIX_TYPES',
    # P3-T5: DAG-GNN
    'DAGGNNEncoder',
    'build_fragmentation_dag',
    'compute_grammar_violation_score',
    'make_empty_dag',
    # P3-T6: CCS
    'SchNetCCSPredictor',
    'JointScorer',
    'CCSLoss',
    'build_ccs_features',
]
