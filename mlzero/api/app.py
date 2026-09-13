from fastapi import FastAPI

from mlzero.api.routes import router

app = FastAPI(title="MLZero Agentic AutoML", description="API for MLZero Engine", version="0.1.0")
app.include_router(router)
