"""FastAPI backend for the RNN translator UI.

Serves the static frontend and proxies translation requests to a local
Ollama server, applying the allow-list defined in config.json.

Run with:
    python server.py
or:
    uvicorn server:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    # Read on every request so editing config.json doesn't require a restart.
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ollama_url(cfg: dict) -> str:
    return cfg.get("ollama_url", "http://localhost:11434").rstrip("/")


def _is_allowed(model: str, allowed: list[str]) -> bool:
    if not allowed:
        return True
    if model in allowed:
        return True
    # Allow matching without the ":tag" suffix, e.g. "llama3.2" matches "llama3.2:latest".
    return model.split(":", 1)[0] in allowed


def _filter_models(installed: Iterable[str], allowed: list[str]) -> list[str]:
    return sorted(name for name in installed if _is_allowed(name, allowed))


app = FastAPI(title="RNN Translator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_client_config() -> dict:
    cfg = load_config()
    return {
        "languages": cfg.get("languages", []),
        "defaultSourceLanguage": cfg.get("default_source_language"),
        "defaultTargetLanguage": cfg.get("default_target_language"),
    }


@app.get("/api/models")
async def list_models() -> dict:
    cfg = load_config()
    allowed = cfg.get("allowed_models") or []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{_ollama_url(cfg)}/api/tags")
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}") from exc

    installed = [m["name"] for m in data.get("models", [])]
    return {
        "models": _filter_models(installed, allowed),
        "installed_count": len(installed),
        "allow_listed": bool(allowed),
    }


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    cfg = load_config()
    allowed = cfg.get("allowed_models") or []

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="`model` field is required")
    if not _is_allowed(model, allowed):
        raise HTTPException(status_code=403, detail=f"Model '{model}' is not in the allow list")

    target = f"{_ollama_url(cfg)}/api/chat"

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", target, json=body) as r:
                if r.status_code >= 400:
                    yield await r.aread()
                    return
                async for chunk in r.aiter_raw():
                    yield chunk

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
