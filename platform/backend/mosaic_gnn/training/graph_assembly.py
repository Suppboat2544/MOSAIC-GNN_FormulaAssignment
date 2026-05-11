"""
MOSAIC-GNN: Graph Assembly Pipeline
=====================================
The critical bridge between Phase 1 (data + graph builders) and
Phase 2 (GNN training).

Converts raw DataFrames + graph builder outputs into MosaicSample
objects with all ground-truth label tensors required by the loss:

  correct_mask         — boolean, marks the correct formula candidate
  candidate_to_feature — int, maps each candidate to its parent feature
  candidate_masses     — float, exact mass per candidate
  eclass_labels        — int, elemental class index per candidate
  dbe_bin_labels       — int, DBE bin index per candidate
  obs_m1 / obs_m2      — float, observed isotope ratios
  iso_feature_mapping  — int, maps ISO sub-graphs to feature indices
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Set
from torch_geometric.data import HeteroData, Data

from mosaic_gnn.data.data_pipeline import (
    load_all_data, get_elemental_class, get_dbe_bin, parse_formula,
    formula_to_mass, compute_dbe,
)
from mosaic_gnn.graphs import (
    build_smdg_vectorized, get_etd_for_sample, detect_elements_from_formulas,
    build_iso_graph, build_adc_graph, build_fcl_for_sample,
)
from mosaic_gnn.graphs.fcl_builder import _formulas_match
from mosaic_gnn.models.hierarchical_head import ECLASS_TO_IDX
from mosaic_gnn.models.film_conditioning import infer_matrix_type
from mosaic_gnn.training.dataset import MosaicSample, MosaicDataset

logger = logging.getLogger(__name__)


def build_mosaic_sample(
    sample_id: str,
    df: pd.DataFrame,
    ppm_error: float = 3.0,
    max_candidates: int = 500,
    element_set: str = 'DOM',
    ion_mode: str = 'negative',
    polarity: str = 'negative',
) -> Optional[MosaicSample]:
    """
    Build a complete MosaicSample from a raw DataFrame.

    This is the critical assembly function that bridges Phase 1 → Phase 2.

    Args:
        sample_id:      Unique identifier for the sample.
        df:             DataFrame with columns: mz_exp, formula, [intensity].
        ppm_error:      Mass accuracy in ppm.
        max_candidates: Max formula candidates per feature.
        element_set:    Element set for candidate enumeration.
        ion_mode:       'negative' or 'positive'.
        polarity:       Adduct polarity for ADC graph.

    Returns:
        MosaicSample with all tensors populated, or None if building fails.
    """
    try:
        # ── Extract arrays ──
        mz_array = df['mz_exp'].values.astype(np.float64)  # keep float64 for mass precision in builders
        N = len(mz_array)

        if N == 0:
            logger.warning(f"  [{sample_id}] Empty sample, skipping")
            return None

        # Intensity: use if available, else synthetic
        if 'intensity' in df.columns:
            intensity_array = df['intensity'].values.astype(np.float32)
        elif 'Relative Intensity' in df.columns:
            intensity_array = df['Relative Intensity'].values.astype(np.float32)
        else:
            intensity_array = np.ones(N, dtype=np.float32)

        # Ground-truth formulas (if available)
        has_formula = 'formula' in df.columns and df['formula'].notna().all()
        true_formulas = df['formula'].astype(str).tolist() if has_formula else None

        logger.info(f"  [{sample_id}] Building MosaicSample: {N} features, "
                     f"has_formula={has_formula}")

        # ── 1. SMDG Graph ──
        formulas_for_etd = true_formulas if has_formula else None
        smdg_data = build_smdg_vectorized(
            mz_array, intensity_array,
            ppm_error=ppm_error,
            formulas=formulas_for_etd,
        )

        # ── 2. ISO Graphs ──
        iso_graphs_all, iso_stats = build_iso_graph(
            mz_array, intensity_array, ppm_error=ppm_error,
        )

        # Build iso_feature_mapping: which features have ISO sub-graphs
        # The iso_builder returns one Data per feature (N total), but we
        # only want ones with actual isotopologues (>1 node)
        iso_graphs = []
        iso_feature_indices = []
        obs_m1_list = []
        obs_m2_list = []

        for g_idx, g in enumerate(iso_graphs_all):
            if g.num_nodes > 1:  # has isotopologues
                iso_graphs.append(g)
                iso_feature_indices.append(g_idx)
                # Extract observed M+1 ratio from the graph's node features
                # Node 0 = monoisotopic, subsequent nodes have observed_ratio
                obs_ratios = g.x[:, 2]  # observed_ratio column
                # M+1 is usually node 1 (if it exists)
                obs_m1 = obs_ratios[1].item() if g.num_nodes > 1 else 0.0
                # M+2 is node 2+ (take max if multiple M+2 types found)
                obs_m2 = obs_ratios[2:].max().item() if g.num_nodes > 2 else 0.0
                obs_m1_list.append(obs_m1)
                obs_m2_list.append(obs_m2)

        iso_feature_mapping = torch.tensor(iso_feature_indices, dtype=torch.long) \
            if iso_feature_indices else torch.zeros(0, dtype=torch.long)
        obs_m1_tensor = torch.tensor(obs_m1_list, dtype=torch.float32) \
            if obs_m1_list else torch.zeros(0)
        obs_m2_tensor = torch.tensor(obs_m2_list, dtype=torch.float32) \
            if obs_m2_list else torch.zeros(0)

        # If no iso graphs with isotopologues, use a dummy
        if len(iso_graphs) == 0:
            dummy_iso = Data(
                x=torch.zeros(1, 4),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
            iso_graphs = [dummy_iso]

        # ── 3. ADC Graph ──
        adc_data = build_adc_graph(
            mz_array, intensity_array,
            ppm_error=ppm_error, polarity=polarity,
        )
        adc_edge_index = adc_data.edge_index  # [2, E_adc]

        # ── 4. FCL Graphs + Ground-Truth Labels ──
        fcl_graphs, fcl_stats = build_fcl_for_sample(
            mz_array, ppm_tolerance=ppm_error,
            element_set=element_set, max_candidates=max_candidates,
            ion_mode=ion_mode, true_formulas=true_formulas,
            inject_formulas=true_formulas,  # ensure ground truth is in candidate set
        )

        # Now compute the label tensors from FCL graphs
        all_correct_mask = []
        all_candidate_to_feature = []
        all_candidate_masses = []
        all_eclass_labels = []
        all_dbe_bin_labels = []

        for feat_idx, fcl_g in enumerate(fcl_graphs):
            n_cand = fcl_g.num_nodes
            if n_cand == 0:
                continue

            # candidate_to_feature: which feature this candidate belongs to
            all_candidate_to_feature.extend([feat_idx] * n_cand)

            # candidate_masses: exact mass from feature 7 (normalized by /1000)
            masses = fcl_g.x[:, 7].numpy() * 1000.0  # un-normalize
            all_candidate_masses.extend(masses.tolist())

            # elemental_class and dbe_bin from the FCL graph metadata
            if hasattr(fcl_g, 'elemental_classes'):
                for ec in fcl_g.elemental_classes:
                    all_eclass_labels.append(ECLASS_TO_IDX.get(ec, 6))  # 6='other'
            else:
                all_eclass_labels.extend([6] * n_cand)

            if hasattr(fcl_g, 'dbe_bins'):
                all_dbe_bin_labels.extend(fcl_g.dbe_bins)
            else:
                all_dbe_bin_labels.extend([0] * n_cand)

            # correct_mask: does this candidate match the ground-truth formula?
            # Use order-independent comparison (handles Hill vs MS notation)
            if has_formula and feat_idx < len(true_formulas):
                true_f = true_formulas[feat_idx]
                if hasattr(fcl_g, 'formula_labels'):
                    for f_label in fcl_g.formula_labels:
                        all_correct_mask.append(_formulas_match(f_label, true_f))
                else:
                    all_correct_mask.extend([False] * n_cand)
            else:
                all_correct_mask.extend([False] * n_cand)

        total_candidates = len(all_correct_mask)

        correct_mask = torch.tensor(all_correct_mask, dtype=torch.bool)
        candidate_to_feature = torch.tensor(all_candidate_to_feature, dtype=torch.long)
        candidate_masses = torch.tensor(all_candidate_masses, dtype=torch.float32)
        eclass_labels = torch.tensor(all_eclass_labels, dtype=torch.long)
        dbe_bin_labels = torch.tensor(all_dbe_bin_labels, dtype=torch.long)
        feature_mz = torch.tensor(mz_array, dtype=torch.float32)

        n_correct = correct_mask.sum().item()
        logger.info(f"  [{sample_id}] Labels: {total_candidates} candidates, "
                     f"{n_correct}/{N} correct found ({n_correct/max(N,1):.1%} recall), "
                     f"{len(iso_graphs)} ISO graphs, {adc_edge_index.shape[1]//2} ADC edges")

        return MosaicSample(
            sample_id=sample_id,
            smdg_data=smdg_data,
            iso_graphs=iso_graphs,
            fcl_graphs=fcl_graphs,
            adc_edge_index=adc_edge_index,
            feature_mz=feature_mz,
            correct_mask=correct_mask,
            candidate_to_feature=candidate_to_feature,
            candidate_masses=candidate_masses,
            eclass_labels=eclass_labels,
            dbe_bin_labels=dbe_bin_labels,
            obs_m1=obs_m1_tensor,
            obs_m2=obs_m2_tensor,
            iso_feature_mapping=iso_feature_mapping,
            matrix_type_idx=infer_matrix_type(sample_id),
        )

    except Exception as e:
        logger.error(f"  [{sample_id}] Failed to build MosaicSample: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_dataset_from_data(
    datasets: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    ppm_error: float = 3.0,
    max_candidates: int = 500,
    element_set: str = 'DOM',
    train_sources: Optional[List[str]] = None,
    val_sources: Optional[List[str]] = None,
    test_sources: Optional[List[str]] = None,
) -> Tuple[MosaicDataset, MosaicDataset, MosaicDataset]:
    """
    Build train/val/test MosaicDatasets from loaded data.

    Default split:
        train: DOM_training_ver2 + DOM_training_ver3_21T
        val:   20% of training (random)
        test:  DOM_testing

    Args:
        datasets:       Dict[sample_id → DataFrame] from load_all_data.
        metadata:       Metadata DataFrame from load_all_data.
        ppm_error:      Mass accuracy in ppm.
        max_candidates: Max candidates per feature.
        element_set:    Element set for enumeration.
        train_sources:  Source labels for training set.
        val_sources:    Source labels for validation set.
        test_sources:   Source labels for test set.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if train_sources is None:
        train_sources = ['DOM_training_ver2', 'DOM_training_ver3_21T']
    if test_sources is None:
        test_sources = ['DOM_testing']

    # Filter samples with formulas only
    labeled = metadata[metadata['has_formula'] == True]

    # Split by source
    train_ids = labeled[labeled['source'].isin(train_sources)]['sample_id'].tolist()
    test_ids = labeled[labeled['source'].isin(test_sources)]['sample_id'].tolist()

    # Val split: if val_sources provided, use them; else take 20% of train
    if val_sources is not None:
        val_ids = labeled[labeled['source'].isin(val_sources)]['sample_id'].tolist()
    else:
        np.random.seed(42)
        n_val = max(1, int(len(train_ids) * 0.2))
        np.random.shuffle(train_ids)
        val_ids = train_ids[:n_val]
        train_ids = train_ids[n_val:]

    logger.info(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Build samples
    def _build_samples(sample_ids: List[str]) -> List[MosaicSample]:
        samples = []
        for sid in sample_ids:
            if sid not in datasets:
                logger.warning(f"  Sample '{sid}' not found in datasets, skipping")
                continue
            df = datasets[sid]
            sample = build_mosaic_sample(
                sample_id=sid, df=df,
                ppm_error=ppm_error,
                max_candidates=max_candidates,
                element_set=element_set,
            )
            if sample is not None:
                samples.append(sample)
        return samples

    logger.info("\n── Building training samples ──")
    train_samples = _build_samples(train_ids)
    logger.info(f"\n── Building validation samples ──")
    val_samples = _build_samples(val_ids)
    logger.info(f"\n── Building test samples ──")
    test_samples = _build_samples(test_ids)

    train_dataset = MosaicDataset(train_samples)
    val_dataset = MosaicDataset(val_samples)
    test_dataset = MosaicDataset(test_samples)

    logger.info(f"\nFinal datasets: train={len(train_dataset)}, "
                f"val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def save_dataset(dataset: MosaicDataset, path: str):
    """Save a MosaicDataset to disk via torch.save."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dataset, path)
    logger.info(f"  Saved dataset to {path} ({len(dataset)} samples)")


def load_dataset(path: str) -> MosaicDataset:
    """Load a MosaicDataset from disk."""
    dataset = torch.load(path, weights_only=False)
    logger.info(f"  Loaded dataset from {path} ({len(dataset)} samples)")
    return dataset
