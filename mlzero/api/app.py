from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mlzero.api.routes import router

app = FastAPI(
    title="MLZero Agentic AutoML",
    description="API for MLZero Engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# ── Serve the new HTML frontend SPA at the root ──────────────────────────
_STATIC_DIR = Path(__file__).parent.parent / "ui" / "static"
_INDEX = _STATIC_DIR / "index.html"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the single-page frontend application."""
    if _INDEX.exists():
        return FileResponse(_INDEX)
    # Fallback if frontend hasn't been built
    from fastapi.responses import JSONResponse
    return JSONResponse({"message": "MLZero API is running. Frontend not found."})  # type: ignore[return-value]
