from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Voice Processor"
    ASR_BACKEND: str = "whisper"
    WHISPER_MODEL: str = "large"
    QWEN_MODEL: str = "Qwen/Qwen3-ASR-0.6B"
    QWEN_LOAD_BACKEND: str = "transformers"
    QWEN_DEVICE_MAP: str = "auto"
    QWEN_DTYPE: str = "bfloat16"
    QWEN_MAX_NEW_TOKENS: int = 256
    QWEN_MAX_INFERENCE_BATCH_SIZE: int = 1
    DEFAULT_LANGUAGE: str = "ja"

    class Config:
        env_file = ".env"

settings = Settings()
