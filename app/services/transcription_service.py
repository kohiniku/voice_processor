import os
import shutil
import tempfile
from fastapi import UploadFile
from app.core.config import settings


LANGUAGE_CODE_MAP = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


class TranscriptionService:
    def __init__(self):
        self.backend = settings.ASR_BACKEND.lower().strip()
        self.model = self._load_model()

    def _load_model(self):
        if self.backend == "whisper":
            import whisper

            model = whisper.load_model(settings.WHISPER_MODEL)
            print(f"Whisper model loaded on device: {model.device}")
            return model

        if self.backend == "qwen":
            try:
                from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
            except ImportError as exc:
                raise RuntimeError(
                    "ASR_BACKEND=qwen is set but qwen-asr is not installed. "
                    "Install it with: pip install qwen-asr"
                ) from exc

            if settings.QWEN_LOAD_BACKEND.lower().strip() != "transformers":
                raise ValueError(
                    "qwen-asr currently supports only QWEN_LOAD_BACKEND=transformers"
                )

            model = Qwen3ASRModel.from_pretrained(
                settings.QWEN_MODEL,
                trust_remote_code=True,
                device_map=settings.QWEN_DEVICE_MAP,
                dtype=settings.QWEN_DTYPE,
                max_new_tokens=settings.QWEN_MAX_NEW_TOKENS,
                max_inference_batch_size=settings.QWEN_MAX_INFERENCE_BATCH_SIZE,
            )
            print(f"Qwen ASR model loaded: {settings.QWEN_MODEL}")
            return model

        raise ValueError(
            "Unsupported ASR_BACKEND value. "
            "Use 'whisper' or 'qwen'. "
            f"Current value: {settings.ASR_BACKEND}"
        )

    async def transcribe(self, file: UploadFile, language: str = None) -> dict:
        temp_audio_path = None
        try:
            suffix = f".{file.filename.split('.')[-1]}" if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                shutil.copyfileobj(file.file, temp_audio)
                temp_audio_path = temp_audio.name

            lang = language or settings.DEFAULT_LANGUAGE

            if self.backend == "whisper":
                result = self.model.transcribe(temp_audio_path, language=lang)
                return {"text": result["text"]}

            result = self.model.transcribe(
                audio=temp_audio_path,
                language=self._normalize_qwen_language(lang),
            )
            return {"text": self._extract_qwen_text(result)}

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def _normalize_qwen_language(self, language: str | None) -> str | None:
        if not language:
            return None
        normalized = language.strip()
        return LANGUAGE_CODE_MAP.get(normalized.lower(), normalized)

    def _extract_qwen_text(self, result) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            text = result.get("text")
            return text if isinstance(text, str) else str(result)

        if isinstance(result, list):
            texts = []
            for item in result:
                if isinstance(item, str):
                    texts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        texts.append(text)
                        continue
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    texts.append(text_attr)
                    continue
                texts.append(str(item))
            return "\n".join(texts)

        text_attr = getattr(result, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        return str(result)
