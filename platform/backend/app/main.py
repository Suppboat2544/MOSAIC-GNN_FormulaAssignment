from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import upload, predict, results, compare, vankreveLen

app = FastAPI(
    title="MOSAIC-GNN Platform API",
    description="Backend API for MOSAIC-GNN molecular formula assignment platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router,      prefix="/api/v1/upload",   tags=["Upload"])
app.include_router(predict.router,     prefix="/api/v1/predict",  tags=["Predict"])
app.include_router(results.router,     prefix="/api/v1/results",  tags=["Results"])
app.include_router(compare.router,     prefix="/api/v1/compare",  tags=["Compare"])
app.include_router(vankreveLen.router, prefix="/api/v1/vankrevelen", tags=["Van Krevelen"])


@app.get("/health")
def health():
    return {"status": "ok"}
