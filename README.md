# MOSAIC-GNN: Molecular Formula Assignment from MS1 Mass Spectrometry

**Mixture-Optimised Spectral–Adduct–Isotope Co-Inference Graph Neural Network**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

MOSAIC-GNN assigns molecular formulas to features detected in high-resolution mass
spectrometry (HRMS) data **using only MS1 information** — no tandem MS/MS fragmentation
spectra required. It co-infers over four complementary graph representations built
directly from a raw peak list.

---

## Key Results

| Benchmark | Top-1 | Top-5 | MRR |
|---|---|---|---|
| DOM test set (IHSS, *n* = 4,047) | **97.68%** | 99.58% | 0.983 |
| GNPS cross-domain (*n* = 3,235, zero-shot) | **86.40%** | 96.38% | — |
| PFAS transfer learning (*n* = 375) | **78.9%** | 92.0% | — |

MOSAIC-GNN M10 surpasses BUDDY (63.4%) by **+34.3 pp** and SIRIUS 5 with MS2
fragmentation (87.3%) by **+10.4 pp** on the same DOM test set.

---

## Architecture

MOSAIC-GNN integrates five specialised neural modules:

| Module | Graph | Purpose |
|---|---|---|
| SMDG R-GCN | Spectral Mass-Difference Graph | Mixture-level context via elemental transformations |
| ISO GAT | Isotopologue Pattern Graph | Heteroatom estimation from isotope envelopes |
| FCL GIN | Formula Candidate Lattice | Energy-based formula ranking with physics constraints |
| ADC Layer | Adduct Consistency Graph | Cross-feature neutral-mass agreement |
| Hierarchical Head | — | Element class → DBE bin → formula ranking |

Additional features: Evidential uncertainty quantification (Normal-Inverse-Gamma
posteriors), FiLM instrument-type conditioning, domain adversarial training.

---

## Web Platform & Desktop App

A ready-to-use inference interface is provided — **no programming required**.

### Web Platform (Next.js + FastAPI)
```bash
cd platform
docker-compose up
# Open http://localhost:3000
```
- Drag-and-drop mzML / CSV upload
- Interactive per-class histogram and paginated formula tables
- REST API at `http://localhost:8000/docs`

### Desktop Application (Tkinter, no internet required)
```bash
pip install -r platform/backend/requirements.txt
python launch.py
```

---

## Installation

```bash
git clone https://github.com/Suppboat2544/MOSAIC-GNN_FormulaAssignment.git
cd MOSAIC-GNN_FormulaAssignment
python -m venv .venv && source .venv/bin/activate
pip install -r platform/backend/requirements.txt
```

---

## Inference (Python API)

```python
import torch
from mosaic_gnn.models.mosaic_gnn import MOSAICGNN

model = MOSAICGNN(use_evidential=True)
ckpt = torch.load("platform/backend/checkpoints/best.pt",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
```

---

## Project Structure

```
mosaic_gnn/          # Core GNN package (models, graphs, losses, training)
platform/
├── frontend/        # Next.js 14 web interface
├── backend/         # FastAPI inference server + model weights
└── docker-compose.yml
scripts/             # Training & evaluation scripts
configs/             # YAML configuration files
tests/               # pytest unit tests
launch.py            # Desktop application launcher
doc/                 # LaTeX manuscript source
```

---

## Citation

If you use MOSAIC-GNN in your research, please cite:

```bibtex
@article{klabklaydee2026mosaicgnn,
  title   = {{MOSAIC-GNN}: A Multi-Graph Neural Network for High-Accuracy
             Molecular Formula Assignment from {MS1}-Only Mass Spectrometry Data},
  author  = {Klabklaydee, Supaporn and Subsa-Ard, Nopphakorn and Chankhow, Suppanut
             and Darumas, Nattapak and Gu, Yuheng and Fujii, Manabu},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
