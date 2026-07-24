# AquaMind AI backend container.
# Build from the PROJECT ROOT so the `agents/` pipeline packages are included:
#   docker build -f backend/Dockerfile -t aquamind-backend .
#   docker run -p 8000:8000 --env-file .env.testing aquamind-backend
# (.env.testing is the confirmed production LLM configuration.)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# System deps occasionally required by scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install backend deps first (better layer caching). If a root requirements.txt
# exists for the AI pipeline, install it too.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt
COPY requirements.tx[t] /app/requirements.txt
RUN if [ -f /app/requirements.txt ]; then pip install -r /app/requirements.txt; fi

# Copy the project (backend + agents pipeline + artifacts).
COPY . /app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host ${HOST} --port ${PORT}"]
