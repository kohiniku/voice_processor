from __future__ import annotations

import json
import os
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.services.transcription_service import TranscriptionService
from app.services.history_service import HistoryService

router = APIRouter()

_service_instance = None
_history_instance = None


def get_transcription_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService()
    return _service_instance


def get_history_service():
    global _history_instance
    if _history_instance is None:
        _history_instance = HistoryService()
    return _history_instance


# ── helpers ──────────────────────────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── endpoints ────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = None,
    service: TranscriptionService = Depends(get_transcription_service),
    history: HistoryService = Depends(get_history_service),
):
    try:
        result = await service.transcribe(file, language=language)
        # Save to history
        filename = file.filename or "unknown"
        history.add(filename, result["text"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe/stream")
async def transcribe_audio_stream(
    file: UploadFile = File(...),
    language: str | None = None,
    service: TranscriptionService = Depends(get_transcription_service),
    history: HistoryService = Depends(get_history_service),
):
    """Stream transcription progress via Server-Sent Events."""

    # Save uploaded file to a temporary location
    suffix = f".{file.filename.split('.')[-1]}" if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    filename = file.filename or "unknown"

    def _generate():
        texts: list[str] = []
        try:
            # Basic validation
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                yield _sse_event("error", {"detail": "アップロードされた音声データが空です。マイクの権限や録音状態を確認してください。"})
                return

            for chunk_idx, total, partial_text in service.transcribe_chunked(
                temp_path, language=language
            ):
                texts.append(partial_text)
                percent = int((chunk_idx + 1) / total * 100)
                yield _sse_event(
                    "progress",
                    {"chunk": chunk_idx + 1, "total": total, "percent": percent},
                )
                yield _sse_event(
                    "partial",
                    {"chunk": chunk_idx + 1, "text": partial_text},
                )

            merged = "".join(texts)
            history.add(filename, merged)
            yield _sse_event("done", {"text": merged})

        except Exception as exc:
            import logging
            logging.error(f"Error during streaming: {exc}")
            error_msg = str(exc)
            if "ffmpeg/avlib" in error_msg or "CouldntDecodeError" in str(type(exc)):
                error_msg = "音声ファイルの読み込みに失敗しました。対応していない形式か、ファイルが空（0 bytes）または破損している可能性があります。"
            yield _sse_event("error", {"message": error_msg})

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/history")
async def get_history(history: HistoryService = Depends(get_history_service)):
    return history.get_all()


@router.delete("/history/{entry_id}")
async def delete_history(
    entry_id: int,
    history: HistoryService = Depends(get_history_service),
):
    if not history.delete(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"ok": True}

