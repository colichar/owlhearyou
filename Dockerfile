FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip

# ── nemotron (CPU) ────────────────────────────────────────────────────────────
FROM base AS nemotron

RUN pip install --no-cache-dir -e ".[server,nemotron]"

ENV STT_BACKEND=nemotron \
    STT_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ── whisper (CPU) ─────────────────────────────────────────────────────────────
# FROM base AS whisper
# RUN pip install --no-cache-dir -e ".[server,whisper]"
# ENV STT_BACKEND=whisper STT_DEVICE=cpu
# CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ── owl (nemotron + whisper, CPU) ─────────────────────────────────────────────
# FROM base AS owl
# RUN pip install --no-cache-dir -e ".[server,nemotron,whisper]"
# ENV STT_DEVICE=cpu
# CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
