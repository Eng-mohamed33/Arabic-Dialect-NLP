from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.components import load_components, load_normalizer, resolve_dialect_path  # noqa: F401 (re-exported)
from src.normalizer.rules import normalize_to_msa
from src.pipeline import DialectPipeline


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


def create_app(*, pipeline: DialectPipeline | None = None, normalizer: Callable[[str], str] | None = None,
               normalizer_backend: str = "custom", load_from_disk: bool = False) -> FastAPI:
    state: dict[str, Any] = {"pipeline": pipeline, "normalizer": normalizer or normalize_to_msa,
                            "backend": normalizer_backend if normalizer else "rules"}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if load_from_disk:  # load once at startup, not per request
            state["pipeline"], state["normalizer"], state["backend"] = load_components()
        yield

    app = FastAPI(title="Arabic Dialect Toolkit", version="0.3.0", lifespan=lifespan)

    def no_model() -> dict[str, str]:
        return {"error": f"No dialect model found at {resolve_dialect_path()}. "
                         "Run `python -m src.dialect_id.train_eval` to create it."}

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = state["pipeline"]
        return {"status": "ok", "dialect_model": loaded is not None, "normalizer": state["backend"],
                "threshold": loaded.threshold if loaded is not None else None}

    @app.post("/identify")
    def identify(request: TextRequest):
        return no_model() if state["pipeline"] is None else state["pipeline"].identify(request.text)

    @app.post("/normalize")
    def normalize(request: TextRequest):
        return {"msa": state["normalizer"](request.text), "backend": state["backend"]}

    @app.post("/pipeline")
    def run_pipeline(request: TextRequest):
        return no_model() if state["pipeline"] is None else state["pipeline"].run(request.text)

    return app


app = create_app(load_from_disk=True)
