FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip

# ── nemotron-kokoro (CPU) ─────────────────────────────────────────────────────
FROM base AS nemotron-kokoro

RUN pip install --no-cache-dir -e ".[server,nemotron,kokoro]"

ENV STT_BACKEND=nemotron \
    STT_DEVICE=cpu \
    KOKORO_VOICE=af_heart

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ── whisper-kokoro (CPU) ──────────────────────────────────────────────────────
FROM base AS whisper-kokoro

RUN pip install --no-cache-dir -e ".[server,whisper,kokoro]"

ENV STT_BACKEND=whisper \
    STT_DEVICE=cpu \
    WHISPER_MODEL=base \
    WHISPER_LANGUAGE= \
    KOKORO_VOICE=af_heart

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
