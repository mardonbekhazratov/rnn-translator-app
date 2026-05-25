"""FastAPI backend for the RNN translator UI.

Serves the static frontend and runs the locally-trained RNN seq2seq model.
There is no external model server — translation happens in-process via
`rnn_infer`, using checkpoints produced by `train.ipynb`.

Run with:
    python server.py
or:
    uvicorn server:app --reload
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
CONFIG_PATH = ROOT / "config.json"

# Model name shown in the UI for the trained RNN. The actual direction
# (en->uz / uz->en) is chosen from the UI's language selectors.
RNN_MODEL = "rnn"


def _rnn_available() -> bool:
    """True if at least one trained checkpoint exists. Imports torch lazily."""
    try:
        import rnn_infer
        return bool(rnn_infer.available_directions())
    except Exception:
        return False


def load_config() -> dict:
    # Read on every request so editing config.json doesn't require a restart.
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    models = [RNN_MODEL] if _rnn_available() else []
    return {"models": models, "rnn_available": bool(models)}


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    model = body.get("model")
    if model != RNN_MODEL:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. This app only serves '{RNN_MODEL}'.")

    import rnn_infer

    text = _last_user_text(body.get("messages", []))
    source = body.get("source_language")
    target = body.get("target_language")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text to translate")

    try:
        translation = rnn_infer.translate(source, target, text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def stream():
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Stream word-by-word in an NDJSON chat shape the frontend understands.
        tokens = translation.split(" ") if translation else []
        for i, tok in enumerate(tokens):
            content = tok if i == 0 else " " + tok
            line = {"model": RNN_MODEL, "created_at": created,
                    "message": {"role": "assistant", "content": content}, "done": False}
            yield json.dumps(line) + "\n"
        final = {"model": RNN_MODEL, "created_at": created,
                 "message": {"role": "assistant", "content": ""}, "done": True}
        yield json.dumps(final) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
