"""
Inference service — runs MOSAIC-GNN model on uploaded mass-spec data.

Model configs
─────────────
Model9  → checkpoints_p3v3/best.pt   gin_formula_feat_dim=10  smdg_num_relations=11
Model10 → checkpoints_run10/best.pt  gin_formula_feat_dim=14  smdg_num_relations=13
Model11 → checkpoints_run11/best.pt  gin_formula_feat_dim=17  smdg_num_relations=13
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

# ── Python 3.14 compat: cProfile C-extension may be absent; stub it so that
#    torch._dynamo (which does `import cProfile`) doesn't crash at import time.
if "cProfile" not in sys.modules:
    import types as _types
    _cprofile_stub = _types.ModuleType("cProfile")

    class _Profile:  # minimal stub — only needs to be importable
        def __init__(self, *a, **kw): pass
        def enable(self): pass
        def disable(self): pass
        def create_stats(self): pass
        def print_stats(self, *a, **kw): pass
        def dump_stats(self, *a, **kw): pass
        def runcall(self, func, *a, **kw): return func(*a, **kw)
        def __enter__(self): return self
        def __exit__(self, *a): pass

    _cprofile_stub.Profile = _Profile  # type: ignore[attr-defined]
    _cprofile_stub.run = lambda *a, **kw: None  # type: ignore[attr-defined]
    _cprofile_stub.runctx = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["cProfile"] = _cprofile_stub

import numpy as np
import pandas as pd
import torch

# ── Suppress torch_geometric DataPipe name-collision warnings ─────────────────
# torch_geometric tries to register 'batch_graphs' as a DataPipe functional name
# but torchdata may have already registered it, causing a noisy UserWarning.
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Unable to add DataPipe function name.*",
    category=UserWarning,
)

from app.core.config import settings
from app.schemas.predict import PredictRequest, JobStatus

# ── add platform/backend/ to path so its bundled mosaic_gnn is importable ──────
BACKEND_DIR = settings.BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

log = logging.getLogger(__name__)

# In-memory job registry (replace with Redis/DB for multi-worker production)
_jobs: dict[str, dict] = {}

# ── Atomic masses & element helpers ───────────────────────────────────────────
ELEMENTS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I", "Si"]
ECLASS_NAMES = ["CHO", "CHON", "CHOS", "CHONS", "CHOF", "CHONSP", "other"]
PROTON_MASS = 1.00727646677  # Da

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula(f: str) -> dict[str, int]:
    if not isinstance(f, str):
        f = str(f) if f else ""
    counts: dict[str, int] = {}
    for m in _FORMULA_RE.finditer(f or ""):
        e = m.group(1)
        if e:
            counts[e] = counts.get(e, 0) + (int(m.group(2)) if m.group(2) else 1)
    return counts


def _ratios(counts: dict[str, int]) -> dict[str, float]:
    c = counts.get("C", 0) or 1
    return {
        "H_C": counts.get("H", 0) / c,
        "O_C": counts.get("O", 0) / c,
        "N_C": counts.get("N", 0) / c,
        "S_C": counts.get("S", 0) / c,
    }


def _vector_to_formula(vec: Any) -> str:
    parts = []
    for elem, count in zip(ELEMENTS, vec):
        c = int(count)
        if c > 0:
            parts.append(f"{elem}{c}" if c > 1 else elem)
    return "".join(parts)


# ── Device ────────────────────────────────────────────────────────────────────
def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _td(obj: Any, device: torch.device) -> Any:
    """Move tensor / PyG object to device, casting float64 → float32."""
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
                    if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
                        store[k] = v.float()
        return obj.to(device)
    return obj


# ── Model loading ─────────────────────────────────────────────────────────────
_MODEL_CONFIGS = {
    "Model9": dict(
        gin_formula_feat_dim=10,
        smdg_num_relations=11,
    ),
    "Model10": dict(
        gin_formula_feat_dim=14,
        smdg_num_relations=13,
    ),
    "Model11": dict(
        gin_formula_feat_dim=17,
        smdg_num_relations=13,
    ),
    "Model11b": dict(
        gin_formula_feat_dim=17,
        smdg_num_relations=13,
    ),  # PFAS fine-tuned, bypass=True (auto-detected from checkpoint)
}

_CKPT_MAP = {
    "Model9":   settings.CKPT_MODEL9,    # checkpoints/Model9/best.pt
    "Model10":  settings.CKPT_MODEL10,   # checkpoints/Model10/best.pt
    "Model11":  settings.CKPT_MODEL11,   # checkpoints/Model11/best.pt
    "Model11b": settings.CKPT_MODEL11B,  # checkpoints/Model11b/best.pt  (PFAS fine-tuned)
}

# Cache loaded models so they are not reloaded on every request
_model_cache: dict[str, Any] = {}


def _detect_formula_feat_dim(state: dict, has_bypass: bool,
                              smdg_hidden: int = 256, iso_out: int = 9,
                              bypass_proj_dim: int = 128) -> int | None:
    """Infer gin_formula_feat_dim from the checkpoint's input_proj weight shape.

    When bypass=False:  input_proj.weight has shape (hidden, formula_feat_dim + smdg_hidden + iso_out)
    When bypass=True:   formula features are first projected to bypass_proj_dim, so
                        input_proj.weight has shape (hidden, bypass_proj_dim + smdg_hidden + iso_out)
                        — formula_feat_dim must be read from formula_proj instead.
    """
    w = state.get("gin_scorer.input_proj.weight")
    if w is None:
        return None
    in_dim = w.shape[1]
    if has_bypass:
        # With bypass the formula portion is always bypass_proj_dim; formula_feat_dim
        # is embedded in formula_proj, not directly visible from input_proj.
        return None
    return in_dim - smdg_hidden - iso_out


def _load_model(model_name: str) -> Any:
    if model_name in _model_cache:
        return _model_cache[model_name]

    from mosaic_gnn.models.mosaic_gnn import MOSAICGNN

    cfg = _MODEL_CONFIGS[model_name]
    ckpt_path = BACKEND_DIR / _CKPT_MAP[model_name]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    has_bypass = "gin_scorer.formula_proj.0.weight" in state

    # Auto-detect formula_feat_dim from the checkpoint so the model is always
    # built with the exact dimension the weights expect, regardless of config.
    detected_feat_dim = _detect_formula_feat_dim(state, has_bypass)
    gin_formula_feat_dim = detected_feat_dim if detected_feat_dim is not None \
        else cfg["gin_formula_feat_dim"]
    log.info(
        f"[{model_name}] gin_formula_feat_dim={gin_formula_feat_dim}"
        f" (detected={detected_feat_dim}, config={cfg['gin_formula_feat_dim']})"
    )

    model = MOSAICGNN(
        smdg_in_channels=4, smdg_hidden_dim=256, smdg_num_layers=4,
        smdg_num_bases=4, smdg_num_relations=cfg["smdg_num_relations"], smdg_dropout=0.1,
        iso_in_channels=4, iso_hidden_dim=128, iso_num_layers=3,
        iso_num_heads=8, iso_dropout=0.1, iso_output_dim=9,
        gin_formula_feat_dim=gin_formula_feat_dim,
        gin_hidden_dim=256, gin_num_layers=3,
        gin_eps_init=0.0, gin_dropout=0.1, gin_use_bypass=has_bypass,
        adc_lambda_init=0.1,
        hier_num_eclass=7, hier_num_dbe_bins=4, hier_dropout=0.1,
        use_evidential=True, use_film=(model_name != "Model9"),
        use_domain_adversarial=(model_name != "Model9"),
        use_dag_gnn=False, use_ccs=False,
    )

    own = model.state_dict()
    for k, v in state.items():
        if k in own and v.shape == own[k].shape:
            own[k] = v
    model.load_state_dict(own)
    device = _device()
    model.to(device).eval()
    log.info(f"Loaded {model_name} onto {device}")
    _model_cache[model_name] = model
    return model


# ── FCL feature patching ──────────────────────────────────────────────────────
def _patch_fcl_run10(dataset_or_samples: Any) -> None:
    """Add 4 Bethe-Weizsäcker nuclear features (10→14 dim)."""
    from mosaic_gnn.data.data_pipeline import ATOMIC_MASSES

    N_idx, S_idx, P_idx = ELEMENTS.index("N"), ELEMENTS.index("S"), ELEMENTS.index("P")

    # Accept either a dataset object or a plain list of samples
    samples = getattr(dataset_or_samples, "samples", dataset_or_samples)

    def _bw(counts: dict[str, int]) -> np.ndarray:
        A = sum(counts.values())
        Z = counts.get("C", 0) * 6 + counts.get("H", 0) + counts.get("N", 0) * 7 + \
            counts.get("O", 0) * 8 + counts.get("S", 0) * 16
        N_neu = max(A - Z, 0)
        f1 = A ** (2 / 3) / 100.0
        f2 = (Z * (Z - 1)) / max(A ** (1 / 3), 1) / 100.0
        f3 = ((A - 2 * Z) ** 2) / max(A, 1) / 10.0
        f4 = 1.0 if A % 2 == 0 and Z % 2 == 0 else (-1.0 if A % 2 == 0 else 0.0)
        return np.array([f1, f2, f3, f4], dtype=np.float32)

    for sample in samples:
        fb = sample.fcl_batch
        if fb.x.shape[1] >= 14:
            continue
        n_nodes = fb.x.shape[0]
        bw = torch.zeros(n_nodes, 4, dtype=torch.float32)
        fv = fb.formula_vectors  # [N, 11]
        for i in range(n_nodes):
            counts = {e: int(fv[i, j].item()) for j, e in enumerate(ELEMENTS) if fv[i, j].item() > 0}
            bw[i] = torch.from_numpy(_bw(counts))
        fb.x = torch.cat([fb.x, bw], dim=1)


def _patch_fcl_run11(dataset_or_samples: Any) -> None:
    """Add 3 interaction features on top of 14-dim (14→17 dim)."""
    N_idx, S_idx, P_idx = ELEMENTS.index("N"), ELEMENTS.index("S"), ELEMENTS.index("P")
    samples = getattr(dataset_or_samples, "samples", dataset_or_samples)
    for sample in samples:
        fb = sample.fcl_batch
        cur = fb.x.shape[1]
        if cur >= 17:
            continue
        if cur < 14:
            _patch_fcl_run10([sample])
        fv = fb.formula_vectors
        n = fb.x.shape[0]
        inter = torch.zeros(n, 3, dtype=torch.float32)
        inter[:, 0] = (fv[:, N_idx] * fv[:, S_idx]) / 10.0
        inter[:, 1] = (fv[:, P_idx] > 0).float()
        inter[:, 2] = (fv[:, N_idx] * fv[:, P_idx]) / 5.0
        fb.x = torch.cat([fb.x, inter], dim=1)


# ── File parsing ──────────────────────────────────────────────────────────────
def _read_uploaded_file(
    job_id: str,
    mz_column: "str | None" = None,
    intensity_column: "str | None" = None,
) -> pd.DataFrame:
    """
    Read CSV or mzML from the upload directory.
    Normalises columns to: mz_exp, intensity, [formula].
    If mz_column / intensity_column are given they take priority over auto-detection.
    """
    upload_dir = settings.UPLOAD_DIR / job_id
    files = list(upload_dir.iterdir())
    if not files:
        raise FileNotFoundError(f"No file found for job {job_id}")

    fpath = files[0]
    suffix = fpath.suffix.lower()

    # ── Column name normalisation helper ─────────────────────────────────────
    def _norm_df(raw: pd.DataFrame) -> pd.DataFrame:
        # Explicit user-specified columns take top priority
        if mz_column and mz_column in raw.columns:
            raw = raw.rename(columns={mz_column: "mz_exp"})
        if intensity_column and intensity_column in raw.columns and "intensity" not in raw.columns:
            raw = raw.rename(columns={intensity_column: "intensity"})

        # Auto-detect m/z if not yet resolved
        if "mz_exp" not in raw.columns:
            remap = {
                "m/z exp.": "mz_exp", "m/z Exp.": "mz_exp", "m/z": "mz_exp",
                "mz": "mz_exp", "mass": "mz_exp", "Mass": "mz_exp",
                "pepmass": "mz_exp", "precursorMz": "mz_exp",
            }
            raw = raw.rename(columns={k: v for k, v in remap.items() if k in raw.columns})

        # Auto-detect intensity
        if "intensity" not in raw.columns:
            iremap = {
                "Intensity": "intensity", "INTENSITY": "intensity",
                "Rel. Intensity": "intensity", "rel. intensity": "intensity",
            }
            raw = raw.rename(columns={k: v for k, v in iremap.items() if k in raw.columns})

        # Formula column
        fremap = {
            "Chem. Formula": "formula", "Chemical Formula": "formula",
            "formula": "formula", "Formula": "formula",
        }
        raw = raw.rename(columns={k: v for k, v in fremap.items() if k in raw.columns})

        # Last resort: first numeric column → mz_exp
        if "mz_exp" not in raw.columns:
            num_cols = raw.select_dtypes(include="number").columns.tolist()
            if num_cols:
                raw = raw.rename(columns={num_cols[0]: "mz_exp"})

        return raw.dropna(subset=["mz_exp"]).sort_values("mz_exp").reset_index(drop=True)

    if suffix == ".csv":
        df = _norm_df(pd.read_csv(fpath, low_memory=False))

    elif suffix in (".tsv", ".txt"):
        # Try tab-separated, fallback to whitespace
        try:
            raw = pd.read_csv(fpath, sep="\t", low_memory=False)
        except Exception:
            raw = pd.read_csv(fpath, sep=r"\s+", engine="python", low_memory=False)
        df = _norm_df(raw)

    elif suffix in (".xlsx", ".xls"):
        df = _norm_df(pd.read_excel(fpath, engine="openpyxl" if suffix == ".xlsx" else None))

    elif suffix == ".mzml":
        try:
            from pyteomics import mzml as _mzml
            rows = []
            with _mzml.MzML(str(fpath)) as reader:
                for spectrum in reader:
                    if spectrum.get("ms level", 1) != 1:
                        continue
                    for mz, intensity in zip(spectrum["m/z array"], spectrum["intensity array"]):
                        if intensity > 0:
                            rows.append({"mz_exp": float(mz), "intensity": float(intensity)})
            df = pd.DataFrame(rows).sort_values("mz_exp").reset_index(drop=True)
        except ImportError:
            raise ImportError("Install pyteomics to parse mzML: pip install pyteomics")

    elif suffix == ".mzxml":
        try:
            from pyteomics import mzxml as _mzxml
            rows = []
            with _mzxml.MzXML(str(fpath)) as reader:
                for spectrum in reader:
                    if spectrum.get("msLevel", 1) != 1:
                        continue
                    for mz, intensity in zip(spectrum["m/z array"], spectrum["intensity array"]):
                        if intensity > 0:
                            rows.append({"mz_exp": float(mz), "intensity": float(intensity)})
            df = pd.DataFrame(rows).sort_values("mz_exp").reset_index(drop=True)
        except ImportError:
            raise ImportError("Install pyteomics to parse mzXML: pip install pyteomics")

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return df


# ── Core inference ────────────────────────────────────────────────────────────
@torch.no_grad()
def _infer_sample(model: Any, sample: Any, device: torch.device) -> list[dict]:
    """Run forward pass on one MosaicSample and return per-feature results."""
    from mosaic_gnn.training.dataset import mosaic_collate_fn

    batch = mosaic_collate_fn([sample])

    outputs = model(
        smdg_data=_td(batch["smdg_data"], device),
        iso_batch=_td(batch["iso_batch"], device),
        fcl_batch=_td(batch["fcl_batch"], device),
        adc_edge_index=_td(batch.get("adc_edge_index"), device),
        feature_mz=_td(batch.get("feature_mz"), device),
        candidate_masses=_td(batch.get("candidate_masses"), device),
        candidate_to_feature=_td(batch.get("candidate_to_feature"), device),
        candidate_eclass=_td(batch.get("eclass_labels"), device),
        candidate_dbe_bin=_td(batch.get("dbe_bin_labels"), device),
        iso_feature_mapping=_td(batch.get("iso_feature_mapping"), device),
        matrix_type_idx=_td(batch.get("matrix_type_idx"), device),
    )

    energies = outputs["corrected_energies"].cpu()
    u_epi = outputs.get("u_epi", torch.zeros_like(energies)).cpu()
    c2f: torch.Tensor = batch["candidate_to_feature"]
    feature_mz: torch.Tensor | None = batch.get("feature_mz")
    candidate_masses: torch.Tensor | None = batch.get("candidate_masses")
    eclass_labels: torch.Tensor | None = batch.get("eclass_labels")

    # formula strings
    fcl_batch = batch["fcl_batch"]
    _raw_labels = getattr(fcl_batch, "formula_labels", None)
    # Batch.from_data_list collates per-graph list[str] into list[list[str]]; flatten it.
    if _raw_labels is not None and _raw_labels and isinstance(_raw_labels[0], list):
        formula_labels: list[str] | None = [s for sub in _raw_labels for s in sub]
    else:
        formula_labels = _raw_labels
    formula_vectors: torch.Tensor | None = getattr(fcl_batch, "formula_vectors", None)

    results = []
    for feat_idx in c2f.unique().tolist():
        mask = c2f == feat_idx
        e_feat = energies[mask]
        order = e_feat.argsort()

        # Top-1 global candidate index
        global_indices = mask.nonzero(as_tuple=False).squeeze(1)
        best_global = global_indices[order[0]].item()

        # Formula
        formula = ""
        if formula_labels and best_global < len(formula_labels):
            formula = formula_labels[best_global]
        elif formula_vectors is not None and best_global < formula_vectors.shape[0]:
            formula = _vector_to_formula(formula_vectors[best_global])

        # Mass error — compute before confidence so accuracy can inform it
        obs_mz = feature_mz[feat_idx].item() if feature_mz is not None else None
        pred_mass = candidate_masses[best_global].item() if candidate_masses is not None else None
        mass_error_ppm: float | None = None
        if pred_mass is not None and obs_mz is not None and obs_mz > 0:
            pred_mz = pred_mass - PROTON_MASS   # [M-H]⁻ adduct
            mass_error_ppm = (pred_mz - obs_mz) / obs_mz * 1e6

        # Confidence: energy-gap score penalised by mass accuracy.
        # Energy gap: large positive gap → the best candidate is clearly
        # separated from the runner-up → high score.
        if len(order) >= 2:
            gap = (e_feat[order[1]] - e_feat[order[0]]).item()
            conf_energy = float(torch.sigmoid(torch.tensor(gap * 5.0)).item())
        else:
            conf_energy = 1.0

        # Mass accuracy penalty: exponentially reduce confidence when the
        # best candidate's predicted m/z is more than 3 ppm from observed.
        # Assignments >10 ppm off collapse to near-zero confidence.
        if mass_error_ppm is not None and abs(mass_error_ppm) > 3.0:
            ppm_excess = abs(mass_error_ppm) - 3.0          # ppm beyond tolerance
            ppm_penalty = math.exp(-0.3 * ppm_excess)       # 10 ppm → 0.12, 20 ppm → 0.01
            confidence = conf_energy * max(0.0, ppm_penalty)
        else:
            confidence = conf_energy

        # Element class
        ec_idx = eclass_labels[best_global].item() if eclass_labels is not None else 6
        ec_name = ECLASS_NAMES[ec_idx] if ec_idx < len(ECLASS_NAMES) else "other"

        # H/C, O/C, N/C, S/C
        counts = _parse_formula(formula)
        ratios = _ratios(counts)

        results.append({
            "mz": round(obs_mz, 6) if obs_mz is not None else None,
            "formula": formula,
            "element_class": ec_name,
            "mass_error_ppm": round(mass_error_ppm, 4) if mass_error_ppm is not None else None,
            "confidence": round(confidence, 4),
            **{k: round(v, 4) for k, v in ratios.items()},
        })

    return results


# ── Public API ────────────────────────────────────────────────────────────────
def get_job_status(job_id: str) -> JobStatus | None:
    rec = _jobs.get(job_id)
    if rec is None:
        return None
    return JobStatus(**rec)


def run_inference(req: PredictRequest) -> None:
    """Background task: load data → build graph → run model → save results."""
    _jobs[req.job_id] = {"job_id": req.job_id, "status": "running", "progress": 0}

    try:
        # If a previous attempt left torch_geometric in a partially-initialised
        # state (circular-import crash), purge it so the fresh import succeeds.
        _tg_mods = [k for k in sys.modules if k.startswith("torch_geometric")]
        for _m in _tg_mods:
            del sys.modules[_m]

        from mosaic_gnn.training.graph_assembly import build_mosaic_sample

        # ── 1. Read uploaded file ─────────────────────────────────────────────
        _jobs[req.job_id]["progress"] = 5
        df = _read_uploaded_file(
            req.job_id,
            mz_column=req.mz_column,
            intensity_column=req.intensity_column,
        )
        log.info(f"[{req.job_id}] Loaded {len(df)} peaks")

        # Apply m/z filter if requested
        if req.mass_range:
            mz_min, mz_max = req.mass_range
            df = df[(df["mz_exp"] >= mz_min) & (df["mz_exp"] <= mz_max)].reset_index(drop=True)

        # ── Sample-type routing ────────────────────────────────────────────────
        # PFAS samples must use Model11b (fine-tuned) and PFAS_AFFF element set.
        # DOM samples use the user-selected model (default: Model10) and DOM set.
        sample_type = getattr(req, "sample_type", None)
        is_pfas = (sample_type is not None and sample_type.value == "PFAS")
        model_name  = "Model11b" if is_pfas else req.model.value
        element_set = "PFAS_AFFF" if is_pfas else "DOM"

        ion_mode = req.ion_mode or "negative"
        if ion_mode == "both":
            ion_mode = "negative"  # default for DOM data

        # ── 2. Build MosaicSample ─────────────────────────────────────────────
        _jobs[req.job_id]["progress"] = 15
        sample = build_mosaic_sample(
            sample_id=req.job_id,
            df=df,
            ppm_error=3.0,
            max_candidates=100,
            element_set=element_set,
            ion_mode=ion_mode,
            polarity=ion_mode,
        )
        if sample is None:
            raise RuntimeError("Failed to build graph from input data")

        # ── 3. Apply model-specific FCL feature patch ─────────────────────────
        _jobs[req.job_id]["progress"] = 30
        if model_name == "Model10":
            _patch_fcl_run10([sample])
        elif model_name in ("Model11", "Model11b"):
            _patch_fcl_run11([sample])

        # ── 4. Load model (cached after first call) ───────────────────────────
        _jobs[req.job_id]["progress"] = 40
        model = _load_model(model_name)
        device = _device()

        # ── 5. Run inference ──────────────────────────────────────────────────
        _jobs[req.job_id]["progress"] = 55
        feature_results = _infer_sample(model, sample, device)

        # ── 6. Save results ───────────────────────────────────────────────────
        _jobs[req.job_id]["progress"] = 90
        out_dir = settings.OUTPUT_DIR / req.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(feature_results, indent=2))

        # Save lightweight comparison stub (real multi-model compare needs all 3 models)
        comparison = {
            "job_id": req.job_id,
            "model_used": model_name,
            "sample_type": element_set,
            "n_features": len(feature_results),
            "metrics": [],  # populated by compare endpoint if user runs all 3 models
        }
        (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

        _jobs[req.job_id].update({"status": "done", "progress": 100})
        log.info(f"[{req.job_id}] Done — {len(feature_results)} features assigned")

    except Exception as exc:
        log.exception(f"[{req.job_id}] Inference failed")
        _jobs[req.job_id].update({"status": "failed", "message": str(exc)})
