# MOSAIC-GNN Platform

Web platform for MOSAIC-GNN molecular formula assignment, built with **Next.js 14** (frontend) and **FastAPI** (backend).

## Features
| Feature | Description |
|---------|-------------|
| Upload | Drag-and-drop CSV / mzML mass-spec files |
| Predict | Run Model9 / Model10 / Model11 with optional m/z range and ion-mode filter |
| Results | Per-mass-range histogram, element-class bar chart, paginated formula table |
| Van Krevelen | Interactive D3 scatter plot (H/C vs O/C), coloured by element class, with hover tooltips |
| Compare | Side-by-side Top-1/Top-5 accuracy bar chart + radar chart for M9/M10/M11 |
| Download | Export results as CSV |

## Quick start (Docker)

```bash
cd platform
docker compose up --build
```

- Frontend: http://localhost:3000  
- API docs: http://localhost:8000/docs

## Local development

**Backend**
```bash
cd platform/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd platform/frontend
npm install
npm run dev
```

## Directory structure

```
platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/   # upload, predict, results, compare, vankrevelen
│   │   ├── core/               # config (settings)
│   │   ├── schemas/            # Pydantic models
│   │   └── services/           # inference, results, compare logic
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/                # Next.js pages: /, /upload, /predict, /results, /compare
│       └── components/
│           ├── charts/         # ElementClassChart, MassRangeHistogram, VanKrevelenPlot, ResultsTable
│           └── layout/         # Sidebar, QueryProvider
└── docker-compose.yml
```

## Integrating MOSAIC-GNN

Edit `backend/app/services/inference.py` and replace the stub with your actual model call:

```python
from mosaic_gnn.predict import run
run(job_id=req.job_id, model=req.model, ...)
```

The `mosaic_gnn/` directory from the parent workspace is mounted read-only at `/app/mosaic_gnn` in the Docker container.
