"""
MOSAIC-GNN CLI entry points.

These are the console_scripts registered by ``pip install mosaic-gnn``.
"""

import sys


def train_cli():
    """Wrapper around ``scripts/run_training_p3.py``."""
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "run_training_p3.py"
    if not script.exists():
        print(f"Training script not found: {script}", file=sys.stderr)
        sys.exit(1)
    # Re-use the existing script's main()
    sys.path.insert(0, str(script.parent))
    import run_training_p3

    run_training_p3.main()


def predict_cli():
    """Minimal predict-from-checkpoint CLI."""
    import argparse
    import torch
    from mosaic_gnn.data.data_pipeline import load_all_data
    from mosaic_gnn.training.graph_assembly import build_dataset_from_data
    from mosaic_gnn.training.dataset import mosaic_collate_fn
    from mosaic_gnn.models.mosaic_gnn import MOSAICGNN

    p = argparse.ArgumentParser(description="MOSAIC-GNN — Predict molecular formulas")
    p.add_argument("--data_root", required=True, help="Path to dataset directory")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--max_candidates", type=int, default=100)
    p.add_argument("--output", default="predictions.json", help="Output JSON path")
    args = p.parse_args()

    import json

    datasets, metadata = load_all_data(args.data_root)
    _, _, test_ds = build_dataset_from_data(
        datasets, metadata, max_candidates=args.max_candidates
    )

    model = MOSAICGNN(use_evidential=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    for k, v in state.items():
        if k in own and v.shape == own[k].shape:
            own[k] = v
    model.load_state_dict(own)
    model.eval()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    predictions = {}
    for sample in test_ds.samples:
        batch = mosaic_collate_fn([sample])

        def _td(obj):
            if obj is None:
                return None
            if isinstance(obj, torch.Tensor):
                if obj.is_floating_point() and obj.dtype != torch.float32:
                    obj = obj.float()
                return obj.to(device)
            if hasattr(obj, "to"):
                if hasattr(obj, "stores"):
                    for store in obj.stores:
                        for k, v in store.items():
                            if (
                                isinstance(v, torch.Tensor)
                                and v.is_floating_point()
                                and v.dtype != torch.float32
                            ):
                                store[k] = v.float()
                return obj.to(device)
            return obj

        with torch.no_grad():
            outputs = model(
                smdg_data=_td(batch["smdg_data"]),
                iso_batch=_td(batch["iso_batch"]),
                fcl_batch=_td(batch["fcl_batch"]),
                adc_edge_index=_td(batch.get("adc_edge_index")),
                feature_mz=_td(batch.get("feature_mz")),
                candidate_masses=_td(batch.get("candidate_masses")),
                candidate_to_feature=_td(batch.get("candidate_to_feature")),
                candidate_eclass=_td(batch.get("eclass_labels")),
                candidate_dbe_bin=_td(batch.get("dbe_bin_labels")),
                iso_feature_mapping=_td(batch.get("iso_feature_mapping")),
            )

        energies = outputs["corrected_energies"].cpu()
        c2f = batch["candidate_to_feature"]
        u_epi = outputs.get("u_epi", torch.zeros_like(energies)).cpu()

        sample_preds = []
        for fi in c2f.unique():
            mask = c2f == fi
            e = energies[mask]
            u = u_epi[mask]
            order = e.argsort()
            sample_preds.append(
                {
                    "feature_idx": fi.item(),
                    "top1_energy": e[order[0]].item(),
                    "top1_uncertainty": u[order[0]].item(),
                }
            )
        predictions[sample.sample_id] = sample_preds

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {args.output}")
