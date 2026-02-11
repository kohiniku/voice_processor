from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.services.transcription_service import TranscriptionService

router = APIRouter()

# Dependency to get the service instance. 
# In a real app, this might be a singleton or lighter dependency inj.
# Since loading the model is heavy, we should simple reuse a global instance or singleton pattern.
# For simplicity here, we'll instantiate it once at module level or use lru_cache.
# However, creating it at module level is the simplest "singleton" in Python.

_service_instance = None

def get_transcription_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService()
    return _service_instance

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = None,
    service: TranscriptionService = Depends(get_transcription_service)
):
    try:
        return await service.transcribe(file, language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
