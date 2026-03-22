import os
import shutil
import tempfile
from collections.abc import Generator
from fastapi import UploadFile
from app.core.config import settings
from app.services.audio_chunker import AudioChunker


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

    async def transcribe(self, file: UploadFile, language: str | None = None) -> dict:
        temp_audio_path = None
        try:
            suffix = f".{file.filename.split('.')[-1]}" if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                shutil.copyfileobj(file.file, temp_audio)
                temp_audio_path = temp_audio.name

            text = self._transcribe_single_file(temp_audio_path, language)
            return {"text": text}

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def transcribe_chunked(
        self, audio_path: str, language: str | None = None
    ) -> Generator[tuple[int, int, str], None, None]:
        """Yield *(chunk_index, total_chunks, partial_text)* for each chunk."""
        chunker = AudioChunker()
        lang = language or settings.DEFAULT_LANGUAGE
        chunks = chunker.chunk(audio_path, settings.CHUNK_DURATION_SECONDS)
        try:
            for chunk in chunks:
                text = self._transcribe_single_file(chunk.path, lang)
                yield chunk.index, chunk.total, text
        finally:
            chunker.cleanup(chunks)

    # ── private helpers ──────────────────────────────────────────

    def _transcribe_single_file(
        self, audio_path: str, language: str | None = None
    ) -> str:
        """Transcribe a single audio file and return the text."""
        lang = language or settings.DEFAULT_LANGUAGE

        if self.backend == "whisper":
            result = self.model.transcribe(audio_path, language=lang)
            return result["text"]

        result = self.model.transcribe(
            audio=audio_path,
            language=self._normalize_qwen_language(lang),
        )
        return self._extract_qwen_text(result)


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
