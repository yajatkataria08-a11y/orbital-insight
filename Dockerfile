# ══════════════════════════════════════════════════════════════════════════════
#  Orbital Insight ACM — Multi-stage Dockerfile
#
#  Base image: ubuntu:22.04  ← REQUIRED by NSH 2026 grading scripts
#
#  Stages:
#    deps      → shared Python dependency layer (cached separately)
#    backend   → FastAPI simulation + ML inference server  (port 8000)
#    streamlit → Analytics dashboard                       (port 8501)
#
#  Build individual targets:
#    docker build --target backend   -t acm-backend   .
#    docker build --target streamlit -t acm-streamlit .
#
#  Or just use docker-compose (recommended):
#    docker-compose up --build
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: shared Python dependency builder ──────────────────────────────────
# ubuntu:22.04 is the MANDATORY base image per NSH 2026 Section 8.
# Ships Python 3.10 by default; we install 3.11 explicitly for consistency.
FROM ubuntu:22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# System packages: Python 3.11 + build tools for scipy / xgboost native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /install

# Copy only requirements first — this layer is cached until requirements change
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: backend (FastAPI + XGBoost inference) ────────────────────────────
FROM ubuntu:22.04 AS backend

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Runtime system deps only (no build-essential needed at serve time)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Reuse installed packages from builder — avoids re-downloading 800 MB of deps
COPY --from=deps /usr/local/lib/python3.11/dist-packages \
                 /usr/local/lib/python3.11/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy the entire backend directory.
# NOTE: training_data.csv (~44 MB) is excluded via .dockerignore — it is only
# needed to retrain, not to serve.  Model .pkl files ARE included.
COPY backend/ .

# Runtime directory for logs, missed_cases.csv, candidate models.
# Mount this as a named volume so artefacts survive container restarts.
RUN mkdir -p /app/runtime \
 && ln -sf /app/runtime/acm.log /app/acm.log

# Non-root user for security
RUN useradd --create-home --shell /bin/bash acm \
 && chown -R acm:acm /app
USER acm

EXPOSE 8000

ENV LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Hits /api/ready — returns 200 once the sim warm-up completes
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/ready')" \
    || exit 1

CMD ["uvicorn", "main:app", \
     "--host",       "0.0.0.0", \
     "--port",       "8000", \
     "--workers",    "1", \
     "--log-level",  "info"]


# ── Stage 3: Streamlit analytics dashboard ─────────────────────────────────────
FROM ubuntu:22.04 AS streamlit

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

COPY --from=deps /usr/local/lib/python3.11/dist-packages \
                 /usr/local/lib/python3.11/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# streamlit + altair are not in requirements.txt
RUN pip install --no-cache-dir \
        "streamlit>=1.35.0" \
        "altair>=5.3.0"

WORKDIR /app



RUN useradd --create-home --shell /bin/bash acm_ui \
 && chown -R acm_ui:acm_ui /app
USER acm_ui

EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]