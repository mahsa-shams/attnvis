# syntax=docker/dockerfile:1

FROM python:3.11-slim

# --- OS deps (minimal) ---
# git: optional (if you pip install from git)
# build-essential: for any wheels that might need compilation (rare but safe)
# libgl1/libglib2.0-0: common deps for opencv / image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# --- Python env defaults ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project (you can override with bind-mount in docker run for dev)
COPY . /app

# Default command: show help (safe)
CMD ["python", "-m", "attnvis", "--help"]

