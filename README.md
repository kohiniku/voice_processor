# Voice Processor

FastAPI based transcription service with switchable ASR backends.

## Supported backends

- `whisper` (default)
- `qwen` (`Qwen/Qwen3-ASR-*` models via `qwen-asr`)

## Environment variables

- `ASR_BACKEND`: `whisper` or `qwen`
- `WHISPER_MODEL`: Whisper model name (for example `large`, `medium`, `small`)
- `QWEN_MODEL`: Hugging Face model id (for example `Qwen/Qwen3-ASR-0.6B`)
- `QWEN_LOAD_BACKEND`: `transformers` (default)
- `QWEN_DEVICE_MAP`: `auto`, `cuda:0`, `cpu`, ...
- `QWEN_DTYPE`: `bfloat16`, `float16`, `float32`, ...
- `QWEN_MAX_NEW_TOKENS`: generation limit for Qwen decoding
- `QWEN_MAX_INFERENCE_BATCH_SIZE`: inference batch size for qwen-asr
- `DEFAULT_LANGUAGE`: default language hint (`ja` is mapped to `Japanese` for Qwen)

## Example (Qwen3-ASR)

Set:

```bash
ASR_BACKEND=qwen
QWEN_MODEL=Qwen/Qwen3-ASR-0.6B
QWEN_DEVICE_MAP=cuda:0
QWEN_DTYPE=bfloat16
DEFAULT_LANGUAGE=ja
```

Then call:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@test.wav"
```
