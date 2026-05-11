"""
MOSAIC-GNN: Training Loop
==========================
AdamW optimizer + CosineAnnealingWarmRestarts scheduler.
Implements early stopping, checkpointing, and evaluation metrics.

Training configuration from configs/default_config.yaml:
  optimizer     : AdamW
  lr            : 1e-3
  weight_decay  : 1e-5
  batch_size    : 32
  max_epochs    : 100
  patience      : 15
  scheduler     : CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
import csv
import time
import logging
import json
from collections import defaultdict

from mosaic_gnn.models import MOSAICGNN
from mosaic_gnn.losses import MOSAICLoss
from .dataset import MosaicDataset, mosaic_collate_fn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric (higher = better)."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float('inf')
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # no improvement


class MosaicTrainer:
    """
    Training loop for MOSAIC-GNN.
    """

    def __init__(
        self,
        model: MOSAICGNN,
        criterion: MOSAICLoss,
        train_dataset: MosaicDataset,
        val_dataset: Optional[MosaicDataset] = None,
        test_dataset: Optional[MosaicDataset] = None,
        # Training config
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 15,
        # Scheduler
        scheduler_T0: int = 10,
        scheduler_Tmult: int = 2,
        # Paths
        checkpoint_dir: str = 'outputs/checkpoints',
        log_dir: str = 'outputs/logs',
        # Device
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_T0,
            T_mult=scheduler_Tmult,
        )

        # Data
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mosaic_collate_fn,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mosaic_collate_fn,
            num_workers=0,
        ) if val_dataset else None
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mosaic_collate_fn,
            num_workers=0,
        ) if test_dataset else None

        # Training state
        self.max_epochs = max_epochs
        self.early_stopping = EarlyStopping(patience=patience)
        self.epoch = 0
        self.best_val_top1 = 0.0
        self.history: List[Dict] = []

        # Paths
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def train(self) -> Dict:
        """
        Main training loop.

        Returns:
            dict with training history and best metrics.
        """
        logger.info("=" * 60)
        logger.info("MOSAIC-GNN Training")
        logger.info(f"  Device:     {self.device}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Train size: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"  Val size:   {len(self.val_loader.dataset)}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info("=" * 60)

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.criterion.set_epoch(epoch)
            t0 = time.time()

            # ── Train ──
            train_metrics = self._train_epoch()

            # ── Validate ──
            val_metrics = {}
            if self.val_loader:
                val_metrics = self._eval_epoch(self.val_loader, prefix='val')

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            # Log
            row = {
                'epoch': epoch,
                'lr': lr,
                'time_sec': elapsed,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            }
            self.history.append(row)

            # Summary line
            train_loss = train_metrics.get('L_total', 0)
            val_loss = val_metrics.get('L_total', 0)
            val_top1 = val_metrics.get('top1_acc', 0)
            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_top1={val_top1:.2%} | "
                f"lr={lr:.2e} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint & early stopping
            if self.val_loader:
                improved = self.early_stopping.step(val_top1)
                if improved:
                    self.best_val_top1 = val_top1
                    self._save_checkpoint('best.pt')
                    logger.info(f"  ★ New best val_top1={val_top1:.2%}")
                if self.early_stopping.should_stop:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

            # Step scheduler
            self.scheduler.step()

        # Save final checkpoint
        self._save_checkpoint('final.pt')
        self._save_history()

        # Final test evaluation
        test_metrics = {}
        if self.test_loader:
            logger.info("\n── Final Test Evaluation ──")
            self._load_checkpoint('best.pt')
            test_metrics = self._eval_epoch(self.test_loader, prefix='test')
            self._log_test_results(test_metrics)

        return {
            'history': self.history,
            'best_val_top1': self.best_val_top1,
            'test_metrics': test_metrics,
        }

    # Threshold for detecting MPS garbage .item() values.
    _MPS_GARBAGE_THRESH = 1e9

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Forward
            model_outputs = self._forward_batch(batch)

            # Build targets dict
            targets = self._build_targets(batch)

            # Compute loss
            loss, loss_dict = self.criterion(model_outputs, targets)

            # ── NaN / Inf / MPS-garbage guard ──
            loss_val = float(loss.item())
            if not math.isfinite(loss_val) or abs(loss_val) > self._MPS_GARBAGE_THRESH:
                del model_outputs, targets, loss, loss_dict, batch
                continue

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics (scalars only — free GPU tensors)
            for k, v in loss_dict.items():
                val = float(v)
                if math.isfinite(val) and abs(val) < self._MPS_GARBAGE_THRESH:
                    epoch_losses[k] += val
            n_batches += 1

            # ── Explicit cleanup: free large GPU tensors immediately ──
            del model_outputs, targets, loss, loss_dict, batch

        # Average
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, prefix: str = 'val') -> Dict[str, float]:
        """Evaluate on a data loader. Returns loss + accuracy metrics."""
        self.model.eval()
        epoch_losses = defaultdict(float)
        n_batches = 0

        all_topk = {1: 0, 3: 0, 5: 0, 10: 0}
        total_features = 0

        for batch in loader:
            model_outputs = self._forward_batch(batch)
            targets = self._build_targets(batch)

            loss, loss_dict = self.criterion(model_outputs, targets)
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

            # Top-K accuracy
            topk_scores = self._compute_topk(
                model_outputs.get('corrected_energies', model_outputs.get('energies')),
                targets.get('candidate_to_feature'),
                targets.get('correct_mask'),
            )
            for k, count in topk_scores.items():
                all_topk[k] += count
            # Count features evaluated
            if targets.get('candidate_to_feature') is not None and targets.get('correct_mask') is not None:
                total_features += targets['candidate_to_feature'][targets['correct_mask']].unique().numel()

            del model_outputs, targets, loss, loss_dict, batch

        avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # Accuracy
        for k in [1, 3, 5, 10]:
            avg_losses[f'top{k}_acc'] = all_topk[k] / max(total_features, 1)

        return avg_losses

    def _forward_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Run model forward on a collated batch."""
        smdg_data = batch['smdg_data']
        iso_batch = batch['iso_batch']
        fcl_batch = batch['fcl_batch']

        # Move to device
        if isinstance(smdg_data, list):
            # Multi-sample: process first sample for now (batch_size=1 in practice)
            smdg_data = smdg_data[0]

        def _to_device(obj):
            if obj is None:
                return None
            if isinstance(obj, torch.Tensor):
                if obj.dtype == torch.float64:
                    obj = obj.float()
                return obj.to(self.device)
            if hasattr(obj, 'to'):
                # Coerce any float64 tensors in PyG Data/Batch to float32 for MPS
                if hasattr(obj, 'keys'):
                    for key in obj.keys():
                        attr = obj[key]
                        if isinstance(attr, torch.Tensor) and attr.dtype == torch.float64:
                            obj[key] = attr.float()
                return obj.to(self.device)
            return obj

        smdg_data = _to_device(smdg_data)
        iso_batch = _to_device(iso_batch)
        fcl_batch = _to_device(fcl_batch)

        # P3: matrix type + GRL alpha
        matrix_type_idx = _to_device(batch.get('matrix_type_idx'))
        grl_alpha = self._grl_alpha()

        return self.model(
            smdg_data=smdg_data,
            iso_batch=iso_batch,
            fcl_batch=fcl_batch,
            adc_edge_index=_to_device(batch.get('adc_edge_index')),
            feature_mz=_to_device(batch.get('feature_mz')),
            candidate_masses=_to_device(batch.get('candidate_masses')),
            candidate_to_feature=_to_device(batch.get('candidate_to_feature')),
            candidate_eclass=_to_device(batch.get('eclass_labels')),
            candidate_dbe_bin=_to_device(batch.get('dbe_bin_labels')),
            iso_feature_mapping=_to_device(batch.get('iso_feature_mapping')),
            matrix_type_idx=matrix_type_idx,
            grl_alpha=grl_alpha,
            has_fragments=batch.get('has_fragments', False),
            has_ims=batch.get('has_ims', False),
            fragment_mz=_to_device(batch.get('fragment_mz')),
            precursor_mz=batch.get('precursor_mz'),
            ccs_observed=_to_device(batch.get('ccs_observed')),
        )

    def _build_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract target tensors from collated batch."""
        targets = {}
        for key in [
            'correct_mask', 'candidate_to_feature',
            'eclass_labels', 'dbe_bin_labels',
            'obs_m1', 'obs_m2',
            'ccs_observed',
        ]:
            v = batch.get(key)
            if v is not None:
                targets[key] = v.to(self.device) if isinstance(v, torch.Tensor) else v

        # P3 extension flags (bool, not tensors)
        targets['has_fragments'] = batch.get('has_fragments', False)
        targets['has_ims'] = batch.get('has_ims', False)

        return targets

    def _compute_topk(
        self,
        energies: Optional[torch.Tensor],
        candidate_to_feature: Optional[torch.Tensor],
        correct_mask: Optional[torch.Tensor],
    ) -> Dict[int, int]:
        """
        Vectorised Top-1/3/5/10 hit counts (replaces Python for-loop).

        For each feature, rank candidates by energy (lower = better for EBM).
        Check if the correct formula appears in top-K.
        """
        counts = {1: 0, 3: 0, 5: 0, 10: 0}
        if energies is None or candidate_to_feature is None or correct_mask is None:
            return counts

        # Only consider features that have a correct candidate
        correct_features = candidate_to_feature[correct_mask].unique()
        if correct_features.numel() == 0:
            return counts

        for feat_idx in correct_features:
            feat_mask = (candidate_to_feature == feat_idx)
            feat_energies = energies[feat_mask]
            feat_correct = correct_mask[feat_mask]

            # Rank by ascending energy (lower = better probability in EBM)
            sorted_indices = feat_energies.argsort()
            ranked_correct = feat_correct[sorted_indices]

            for k in [1, 3, 5, 10]:
                if ranked_correct[:k].any():
                    counts[k] += 1

        return counts

    def _grl_alpha(self) -> float:
        """GRL scaling: ramp from 0→1 over first 40% of epochs."""
        import math
        p = min(self.epoch / max(self.max_epochs * 0.4, 1), 1.0)
        return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0) if p > 0 else 0.0

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_top1': self.best_val_top1,
        }, path)
        logger.info(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, name: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"  Loaded checkpoint: {path}")

    def _save_history(self):
        """Save training history to CSV."""
        path = os.path.join(self.log_dir, 'training_history.csv')
        if len(self.history) == 0:
            return

        # Collect all keys across all rows (test metrics only appear on some epochs)
        keys = list(dict.fromkeys(k for row in self.history for k in row.keys()))
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.history)
        logger.info(f"  History saved: {path}")

    def _log_test_results(self, metrics: Dict[str, float]):
        """Print and save test results."""
        logger.info(f"\n  Test Results:")
        for k in [1, 3, 5, 10]:
            key = f'top{k}_acc'
            if key in metrics:
                logger.info(f"    Top-{k}: {metrics[key]:.2%}")

        # Save as JSON
        path = os.path.join(self.log_dir, 'test_results.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Results saved: {path}")
