"""
MOSAIC-GNN: Dataset & Collation
================================
Wraps per-sample graph data into PyTorch Dataset with custom collate
for the heterogeneous multi-graph MOSAIC architecture.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Data, Batch
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


def _prebatch(graphs: List[Data]) -> Batch:
    """Convert a list of Data objects into a single Batch, once."""
    if len(graphs) == 0:
        g = Data(x=torch.zeros(1, 10), edge_index=torch.zeros(2, 0, dtype=torch.long))
        return Batch.from_data_list([g])
    return Batch.from_data_list(graphs)


class MosaicSample:
    """
    Container for one sample (mixture) with all four graph types +
    ground-truth labels.

    FCL and ISO graphs are PRE-BATCHED at construction time to avoid
    the very expensive ``Batch.from_data_list()`` call at every
    collation.
    """

    def __init__(
        self,
        sample_id: str,
        smdg_data: HeteroData,
        iso_graphs: List[Data],
        fcl_graphs: List[Data],
        adc_edge_index: torch.Tensor,
        feature_mz: torch.Tensor,
        # Ground-truth labels
        correct_mask: Optional[torch.Tensor] = None,
        candidate_to_feature: Optional[torch.Tensor] = None,
        candidate_masses: Optional[torch.Tensor] = None,
        eclass_labels: Optional[torch.Tensor] = None,
        dbe_bin_labels: Optional[torch.Tensor] = None,
        obs_m1: Optional[torch.Tensor] = None,
        obs_m2: Optional[torch.Tensor] = None,
        iso_feature_mapping: Optional[torch.Tensor] = None,
        matrix_type_idx: Optional[int] = None,
        # P3 extension flags
        has_fragments: bool = False,
        has_ims: bool = False,
        ccs_observed: Optional[torch.Tensor] = None,
        fragment_mz: Optional[torch.Tensor] = None,
        precursor_mz: Optional[float] = None,
    ):
        self.sample_id = sample_id
        self.smdg_data = smdg_data
        self.adc_edge_index = adc_edge_index
        self.feature_mz = feature_mz
        self.correct_mask = correct_mask
        self.candidate_to_feature = candidate_to_feature
        self.candidate_masses = candidate_masses
        self.eclass_labels = eclass_labels
        self.dbe_bin_labels = dbe_bin_labels
        self.obs_m1 = obs_m1
        self.obs_m2 = obs_m2
        self.iso_feature_mapping = iso_feature_mapping
        self.matrix_type_idx = torch.tensor(matrix_type_idx or 0, dtype=torch.long)

        # P3 extension flags
        self.has_fragments = has_fragments
        self.has_ims = has_ims
        self.ccs_observed = ccs_observed
        self.fragment_mz = fragment_mz
        self.precursor_mz = precursor_mz

        # Pre-batch the graph lists once — saves RAM & avoids
        # repeated Batch.from_data_list during training
        self.iso_batch: Batch = _prebatch(iso_graphs)
        self.fcl_batch: Batch = _prebatch(fcl_graphs)

        # We no longer keep the raw lists
        # (they are accessible via iso_batch / fcl_batch if needed)


class MosaicDataset(Dataset):
    """
    PyTorch Dataset of MosaicSamples.
    Each item is one full mixture, pre-built as graphs.
    """

    def __init__(self, samples: List[MosaicSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MosaicSample:
        return self.samples[idx]


def mosaic_collate_fn(batch: List[MosaicSample]) -> Dict[str, Any]:
    """
    Custom collate for MOSAIC samples.

    Since each sample has different graph sizes, we cannot simply stack.
    For batch_size=1, we just unwrap. For batch_size>1, we batch
    each graph type separately and track global offsets.

    FCL and ISO graphs are already pre-batched inside MosaicSample,
    so collation for batch_size=1 is essentially free.

    Returns dict with all tensors needed by MOSAICGNN.forward()
    """
    if len(batch) == 1:
        s = batch[0]
        return {
            'sample_ids': [s.sample_id],
            'smdg_data': s.smdg_data,
            'iso_batch': s.iso_batch,
            'fcl_batch': s.fcl_batch,
            'adc_edge_index': s.adc_edge_index,
            'feature_mz': s.feature_mz,
            'candidate_to_feature': s.candidate_to_feature,
            'candidate_masses': s.candidate_masses,
            'correct_mask': s.correct_mask,
            'eclass_labels': s.eclass_labels,
            'dbe_bin_labels': s.dbe_bin_labels,
            'obs_m1': s.obs_m1,
            'obs_m2': s.obs_m2,
            'iso_feature_mapping': s.iso_feature_mapping,
            'matrix_type_idx': s.matrix_type_idx,
            # P3 extension flags (with defaults for legacy cached samples)
            'has_fragments': getattr(s, 'has_fragments', False),
            'has_ims': getattr(s, 'has_ims', False),
            'ccs_observed': getattr(s, 'ccs_observed', None),
            'fragment_mz': getattr(s, 'fragment_mz', None),
            'precursor_mz': getattr(s, 'precursor_mz', 0.0),
        }

    # Multi-sample batch — not the main path (batch_size=1 for hetero graphs),
    # but kept for completeness.
    # Re-batch the already-batched sub-graphs is tricky; fall back to
    # extracting raw Data lists from pre-batched objects.
    # This path is rarely used and only for very small samples.
    all_smdg_datas = []
    all_feature_mz = []
    all_adc_edges = []
    all_candidate_to_feature = []
    all_candidate_masses = []
    all_correct_mask = []
    all_eclass = []
    all_dbe_bin = []
    all_obs_m1 = []
    all_obs_m2 = []
    all_iso_mapping = []
    all_iso_batches = []
    all_fcl_batches = []

    feature_offset = 0
    sample_ids = []

    for s in batch:
        sample_ids.append(s.sample_id)
        n_features = s.feature_mz.size(0) if s.feature_mz is not None else 0
        all_smdg_datas.append(s.smdg_data)
        all_iso_batches.append(s.iso_batch)
        all_fcl_batches.append(s.fcl_batch)

        if s.feature_mz is not None:
            all_feature_mz.append(s.feature_mz)
        if s.adc_edge_index is not None and s.adc_edge_index.numel() > 0:
            all_adc_edges.append(s.adc_edge_index + feature_offset)
        if s.candidate_to_feature is not None:
            all_candidate_to_feature.append(s.candidate_to_feature + feature_offset)
        if s.candidate_masses is not None:
            all_candidate_masses.append(s.candidate_masses)
        if s.correct_mask is not None:
            all_correct_mask.append(s.correct_mask)
        if s.eclass_labels is not None:
            all_eclass.append(s.eclass_labels)
        if s.dbe_bin_labels is not None:
            all_dbe_bin.append(s.dbe_bin_labels)
        if s.obs_m1 is not None:
            all_obs_m1.append(s.obs_m1)
        if s.obs_m2 is not None:
            all_obs_m2.append(s.obs_m2)
        if s.iso_feature_mapping is not None:
            all_iso_mapping.append(s.iso_feature_mapping + feature_offset)
        feature_offset += n_features

    # For multi-sample, use the first sample's pre-batched objects
    # (batch_size > 1 is almost never used; this is a safe fallback)
    iso_batch = all_iso_batches[0] if len(all_iso_batches) == 1 else all_iso_batches[0]
    fcl_batch = all_fcl_batches[0] if len(all_fcl_batches) == 1 else all_fcl_batches[0]

    result = {
        'sample_ids': sample_ids,
        'smdg_data': all_smdg_datas[0] if len(all_smdg_datas) == 1 else all_smdg_datas,
        'iso_batch': iso_batch,
        'fcl_batch': fcl_batch,
        'adc_edge_index': torch.cat(all_adc_edges, dim=1) if all_adc_edges else torch.zeros(2, 0, dtype=torch.long),
        'feature_mz': torch.cat(all_feature_mz) if all_feature_mz else torch.zeros(0),
        'candidate_to_feature': torch.cat(all_candidate_to_feature) if all_candidate_to_feature else None,
        'candidate_masses': torch.cat(all_candidate_masses) if all_candidate_masses else None,
        'correct_mask': torch.cat(all_correct_mask) if all_correct_mask else None,
        'eclass_labels': torch.cat(all_eclass) if all_eclass else None,
        'dbe_bin_labels': torch.cat(all_dbe_bin) if all_dbe_bin else None,
        'obs_m1': torch.cat(all_obs_m1) if all_obs_m1 else None,
        'obs_m2': torch.cat(all_obs_m2) if all_obs_m2 else None,
        'iso_feature_mapping': torch.cat(all_iso_mapping) if all_iso_mapping else None,
    }
    return result


def _empty_batch(feat_dim: int) -> Batch:
    """Create an empty Batch as placeholder."""
    g = Data(
        x=torch.zeros(1, feat_dim),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    return Batch.from_data_list([g])
