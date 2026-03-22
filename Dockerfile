FROM python:3.12

WORKDIR /app

# Install system dependencies
# ffmpeg is required by openai-whisper

RUN apt update && apt install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
# We don't have a lock file that we can easily use with pip alone unless we use a tool like uv or poetry.
# Assuming standard pip usage for now based on the simple pyproject.toml.
# Since it's a [project] style toml, we can install it directly if we had a build system, 
# but simply pip installing the packages listed is safer if we don't assume a build backend.
# However, newer pip can parse pyproject.toml. Let's try installing from it.
# Or safer, manually install dependencies or export them.
# The user's environment has `uv.lock`. They are likely using `uv`.
# I should try to use `uv` in the dockerfile if possible, or just pip install the listed dependencies.
# Given the user instruction "docker composeで", and valid pyproject.toml, let's use pip to install.
# Note: "openai-whisper" pulls in torch which is heavy. 
RUN pip install --no-cache-dir fastapi uvicorn python-multipart openai-whisper qwen-asr pydantic-settings pydub

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
