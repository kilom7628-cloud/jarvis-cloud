"""
JARVIS Cloud Server — FastAPI API Proxy
========================================
Deployed on Render.com. Acts as a secure proxy so the local JARVIS desktop
client never needs to hold API keys. All Groq / Fish Audio calls are made
server-side; the client authenticates with a shared JARVIS_SECRET token.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("jarvis-cloud")

# ---------------------------------------------------------------------------
# Configuration (all sourced from environment variables)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
FISH_API_KEY = os.getenv("FISH_API_KEY", "")
FISH_VOICE_ID = os.getenv("FISH_VOICE_ID", "05b36da8574341d0803391491850db20")
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free")
JARVIS_SECRET = os.getenv("JARVIS_SECRET", "")

# Persistent-ish storage paths (ephemeral on Render free tier, but fine)
DATA_DIR = Path(__file__).parent / "data"
MEMORY_FILE = DATA_DIR / "memory.json"
HISTORY_FILE = DATA_DIR / "history.json"

# Groq & Fish endpoints
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
FISH_TTS_URL = "https://api.fish.audio/v1/tts"

# Edge-TTS fallback voice
EDGE_TTS_VOICE = "en-GB-RyanNeural"

# Server start time (for uptime reporting)
START_TIME = time.time()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="JARVIS Cloud Proxy", version="1.0.0")

# CORS — allow all origins (needed for potential web access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    """Ensure data directory and files exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]", encoding="utf-8")
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
    logger.info("JARVIS Cloud Server started — model=%s", GROQ_MODEL)


# ---------------------------------------------------------------------------
# Web UI Endpoints (Access JARVIS from any phone / tablet / browser anywhere)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>JARVIS Cloud Online</h1>")

@app.post("/chat")
async def web_chat(req: Request):
    """Direct chat endpoint for the web UI (used from mobile/tablets/browser)."""
    try:
        data = await req.json()
    except Exception:
        data = {}
    text = data.get("text", "")
    if not text:
        return {"reply": "Yes, sir?"}

    messages = [
        {"role": "system", "content": "You are JARVIS, Milo's witty, polite British AI assistant. Address him as sir. Replies are spoken aloud so keep them concise without markdown or lists."},
        {"role": "user", "content": text}
    ]
    payload = {"model": GROQ_MODEL, "messages": messages}
    if "gpt-oss" in GROQ_MODEL:
        payload["reasoning_effort"] = "low"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(GROQ_CHAT_URL, json=payload, headers=headers)
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"].get("content", "Done, sir.")
            return {"reply": reply}
        return {"reply": "I am experiencing high network traffic, sir."}

@app.post("/tts")
async def web_tts(req: Request):
    """Direct TTS endpoint for the web UI."""
    try:
        data = await req.json()
    except Exception:
        data = {}
    text = data.get("text", "")
    fish_payload = {
        "text": text,
        "reference_id": FISH_VOICE_ID,
        "mp3_bitrate": 128,
        "format": "mp3",
        "latency": "balanced",
        "prosody": {"speed": 0.9, "volume": 0}
    }
    headers = {
        "Authorization": f"Bearer {FISH_API_KEY}",
        "Content-Type": "application/json",
        "model": FISH_MODEL,
        "User-Agent": "jarvis/1.0"
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(FISH_TTS_URL, json=fish_payload, headers=headers)
        if r.status_code == 200:
            return Response(content=r.content, media_type="audio/mpeg")
    return Response(status_code=500)

@app.get("/events")
async def web_events():
    return {"events": [], "last": 0}

@app.post("/cancel")
async def web_cancel():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _verify_auth(authorization: Optional[str]) -> None:
    """Raise 401 if the Bearer token doesn't match JARVIS_SECRET."""
    if not JARVIS_SECRET:
        # If no secret is configured, skip auth (local dev convenience)
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != JARVIS_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    model: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    speed: Optional[float] = 0.9


class VisionRequest(BaseModel):
    question: str
    image_b64: str


class MemoryAction(BaseModel):
    action: str  # "add" or "remove"
    note: Optional[str] = None
    text: Optional[str] = None


class HistoryPayload(BaseModel):
    history: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers — file I/O
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _write_json(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# POST /api/chat — Proxy to Groq chat completions
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(body: ChatRequest, authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)

    model = body.model or GROQ_MODEL

    # Build the payload for Groq
    payload: Dict[str, Any] = {
        "model": model,
        "messages": body.messages,
    }

    # For gpt-oss models, add low reasoning effort
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"

    if body.tools:
        payload["tools"] = body.tools
        payload["tool_choice"] = body.tool_choice or "auto"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Retry logic: 4 attempts, exponential backoff (3s, 6s, 12s)
    backoff_delays = [3, 6, 12]
    max_attempts = 4

    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(max_attempts):
            try:
                resp = await client.post(GROQ_CHAT_URL, json=payload, headers=headers)

                if resp.status_code == 429 and attempt < max_attempts - 1:
                    delay = backoff_delays[attempt]
                    logger.warning(
                        "Groq 429 rate-limited — retrying in %ds (attempt %d/%d)",
                        delay, attempt + 1, max_attempts,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    logger.error("Groq error %d: %s", resp.status_code, resp.text[:500])
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail={"error": "Groq API error", "body": resp.text[:1000]},
                    )

                return resp.json()

            except httpx.TimeoutException:
                logger.error("Groq request timed out (attempt %d/%d)", attempt + 1, max_attempts)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(backoff_delays[attempt])
                    continue
                raise HTTPException(status_code=504, detail="Groq request timed out")

            except HTTPException:
                raise  # re-raise our own HTTPExceptions

            except Exception as exc:
                logger.exception("Unexpected error calling Groq")
                raise HTTPException(status_code=502, detail=str(exc))

    # Should never reach here, but just in case
    raise HTTPException(status_code=502, detail="Failed to get response from Groq")


# ---------------------------------------------------------------------------
# POST /api/tts — Proxy to Fish Audio (with edge-tts fallback)
# ---------------------------------------------------------------------------
@app.post("/api/tts")
async def tts(body: TTSRequest, authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)

    voice_id = body.voice_id or FISH_VOICE_ID
    audio_bytes: Optional[bytes] = None

    # --- Try Fish Audio first ---
    try:
        fish_payload = {
            "text": body.text,
            "reference_id": voice_id,
            "mp3_bitrate": 128,
            "format": "mp3",
            "latency": "balanced",
        }
        if body.speed is not None:
            fish_payload["prosody"] = {"speed": body.speed, "volume": 0}

        headers = {
            "Authorization": f"Bearer {FISH_API_KEY}",
            "Content-Type": "application/json",
            "model": FISH_MODEL,
            "User-Agent": "jarvis/1.0",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(FISH_TTS_URL, json=fish_payload, headers=headers)

        if resp.status_code == 200:
            audio_bytes = resp.content
            logger.info("Fish TTS succeeded — %d bytes", len(audio_bytes))
        else:
            logger.warning("Fish TTS failed (%d): %s", resp.status_code, resp.text[:300])

    except Exception as exc:
        logger.warning("Fish TTS exception: %s", exc)

    # --- Fallback to edge-tts ---
    if audio_bytes is None:
        try:
            import edge_tts

            logger.info("Falling back to edge-tts (%s)", EDGE_TTS_VOICE)
            communicate = edge_tts.Communicate(body.text, EDGE_TTS_VOICE)
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            audio_bytes = b"".join(chunks)
            logger.info("edge-tts succeeded — %d bytes", len(audio_bytes))

        except Exception as exc:
            logger.exception("edge-tts fallback also failed")
            raise HTTPException(status_code=502, detail=f"TTS failed: {exc}")

    return Response(content=audio_bytes, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# POST /api/vision — Vision model via Groq
# ---------------------------------------------------------------------------
@app.post("/api/vision")
async def vision(body: VisionRequest, authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": body.question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{body.image_b64}",
                    },
                },
            ],
        }
    ]

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": messages,
        "max_tokens": 1024,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(GROQ_CHAT_URL, json=payload, headers=headers)

        if resp.status_code >= 400:
            logger.error("Groq vision error %d: %s", resp.status_code, resp.text[:500])
            raise HTTPException(
                status_code=resp.status_code,
                detail={"error": "Groq vision error", "body": resp.text[:1000]},
            )

        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Vision endpoint failed")
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /api/memory — Retrieve memory notes
# ---------------------------------------------------------------------------
@app.get("/api/memory")
async def get_memory(authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)
    return _read_json(MEMORY_FILE)


# ---------------------------------------------------------------------------
# POST /api/memory — Add or remove memory notes
# ---------------------------------------------------------------------------
@app.post("/api/memory")
async def update_memory(body: MemoryAction, authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)

    notes = _read_json(MEMORY_FILE)

    if body.action == "add":
        if not body.note:
            raise HTTPException(status_code=400, detail="Missing 'note' field")
        notes.append(body.note)
        _write_json(MEMORY_FILE, notes)
        logger.info("Memory note added — total: %d", len(notes))
        return {"status": "added", "total": len(notes)}

    elif body.action == "remove":
        if not body.text:
            raise HTTPException(status_code=400, detail="Missing 'text' field")
        # Remove notes that contain the given text (case-insensitive)
        original_len = len(notes)
        notes = [n for n in notes if body.text.lower() not in str(n).lower()]
        _write_json(MEMORY_FILE, notes)
        removed = original_len - len(notes)
        logger.info("Memory notes removed: %d — remaining: %d", removed, len(notes))
        return {"status": "removed", "removed": removed, "total": len(notes)}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")


# ---------------------------------------------------------------------------
# GET /api/history — Retrieve conversation history
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history(authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)
    history = _read_json(HISTORY_FILE)
    # Return last 40 entries
    return history[-40:]


# ---------------------------------------------------------------------------
# POST /api/history — Save conversation history
# ---------------------------------------------------------------------------
@app.post("/api/history")
async def save_history(body: HistoryPayload, authorization: Optional[str] = Header(None)):
    _verify_auth(authorization)
    _write_json(HISTORY_FILE, body.history)
    logger.info("History saved — %d entries", len(body.history))
    return {"status": "saved", "count": len(body.history)}


# ---------------------------------------------------------------------------
# GET /api/health — Public health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "vision_model": GROQ_VISION_MODEL,
        "uptime": round(time.time() - START_TIME, 1),
    }


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
