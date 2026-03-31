# ══════════════════════════════════════════════════════════════════════════════
#  Orbital Insight ACM — Multi-stage Dockerfile
#
#  Base image: ubuntu:22.04  ← MANDATORY per NSH 2026 Section 8
#
#  Python 3.10 ships natively with Ubuntu 22.04 — no PPA required.
#
#  Project structure:
#    backend/        → FastAPI + ML (main.py, .pkl files)
#    frontend/       → Static files (index.html, earth.jpg) served by nginx
#    streamlit_app/  → Streamlit dashboard (app.py)
#
#  Stages:
#    deps      → shared Python dependency layer (cached separately)
#    backend   → FastAPI simulation + ML inference server  (port 8000)
#    streamlit → Analytics dashboard                       (port 8501)
#
#  Frontend (nginx) is handled entirely by docker-compose — no build stage needed.
#
#  Build individual targets:
#    docker build --target backend   -t acm-backend   .
#    docker build --target streamlit -t acm-streamlit .
#
#  Or just use docker-compose (recommended):
#    docker-compose up --build
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: shared Python dependency builder ──────────────────────────────────
FROM ubuntu:22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Python 3.10 is built into Ubuntu 22.04 — no PPA or add-apt-repository needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pip and set python3.10 as default python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /install

# Root requirements.txt (shared backend deps)
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: backend (FastAPI + XGBoost inference) ────────────────────────────
FROM ubuntu:22.04 AS backend

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime system deps only — no build tools needed at serve time
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Reuse installed packages from builder — avoids re-downloading all deps
COPY --from=deps /usr/local/lib/python3.10/dist-packages \
                 /usr/local/lib/python3.10/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy backend source (main.py, .pkl models, feature_names.json, etc.)
COPY backend/ .

# Runtime directory for logs, missed_cases.csv, candidate models.
# Mount as a named volume so artefacts survive container restarts.
RUN mkdir -p /app/runtime \
 && ln -sf /app/runtime/acm.log /app/acm.log

# Non-root user for security
RUN useradd --create-home --shell /bin/bash acm \
 && chown -R acm:acm /app
USER acm

# Port 8000 must be exported — grading scripts hit this directly (NSH Section 8)
EXPOSE 8000

ENV LOG_LEVEL=INFO

# Hits /api/ready — returns 200 once the sim warm-up completes
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/ready')" \
    || exit 1

# Binds to 0.0.0.0 as required by NSH Section 8 (not just localhost)
CMD ["uvicorn", "main:app", \
     "--host",       "0.0.0.0", \
     "--port",       "8000", \
     "--workers",    "1", \
     "--log-level",  "info"]


# ── Stage 3: Streamlit analytics dashboard ─────────────────────────────────────
FROM ubuntu:22.04 AS streamlit

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Reuse shared deps from builder
COPY --from=deps /usr/local/lib/python3.10/dist-packages \
                 /usr/local/lib/python3.10/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Install streamlit-specific deps from streamlit_app/requirements.txt
COPY streamlit_app/requirements.txt ./streamlit_requirements.txt
RUN pip install --no-cache-dir -r streamlit_requirements.txt \
 && pip install --no-cache-dir \
        "streamlit>=1.35.0" \
        "altair>=5.3.0"

WORKDIR /app

# Copy streamlit app source from streamlit_app/ folder
COPY streamlit_app/app.py .

RUN useradd --create-home --shell /bin/bash acm_ui \
 && chown -R acm_ui:acm_ui /app
USER acm_ui

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]