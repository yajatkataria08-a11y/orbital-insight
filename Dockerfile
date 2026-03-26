FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    nginx curl supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── Python deps ────────────────────────────────────────────────────────────────
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir streamlit altair requests

# ── App source ─────────────────────────────────────────────────────────────────
# backend/
COPY backend/main.py          /app/backend/main.py
COPY backend/train_model.py   /app/backend/train_model.py
COPY backend/generate_data.py /app/backend/generate_data.py

# frontend/
COPY frontend/index.html      /var/www/html/index.html
COPY frontend/earth.jpg       /var/www/html/earth.jpg

# streamlit_app/
COPY streamlit_app/app.py     /app/streamlit_app/app.py

# start.sh (root level)
COPY start.sh                 /app/start.sh
RUN chmod +x /app/start.sh

# Model artefacts — copy from backend/ if pre-trained, else volume-mount at runtime
COPY backend/collision_model.pkl  /app/backend/collision_model.pkl
COPY backend/model_features.pkl   /app/backend/model_features.pkl
COPY backend/model_threshold.pkl  /app/backend/model_threshold.pkl
COPY backend/model_meta.json      /app/backend/model_meta.json

# ── Nginx config ───────────────────────────────────────────────────────────────
RUN cat > /etc/nginx/sites-available/default << 'NGINXEOF'
server {
    listen 80;
    server_name _;

    location / {
        root /var/www/html;
        index index.html;
        try_files $uri /index.html;
    }
    location /api/ {
        proxy_pass         http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_read_timeout 300;
        add_header Access-Control-Allow-Origin  *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type";
    }
    location /docs         { proxy_pass http://127.0.0.1:8000/docs; }
    location /openapi.json { proxy_pass http://127.0.0.1:8000/openapi.json; }
}

server {
    listen 8501;
    server_name _;
    location / {
        proxy_pass         http://127.0.0.1:8502/;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host       $host;
        proxy_read_timeout 86400;
    }
}
NGINXEOF

# ── Supervisord config ─────────────────────────────────────────────────────────
RUN mkdir -p /var/log/supervisor

RUN cat > /etc/supervisor/conf.d/orbital.conf << 'SUPEOF'
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor

[unix_http_server]
file=/var/run/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

; ── Nginx ──────────────────────────────────────────────────────────────────────
[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
priority=10
stdout_logfile=/var/log/supervisor/nginx.log
stderr_logfile=/var/log/supervisor/nginx_err.log

; ── FastAPI backend ────────────────────────────────────────────────────────────
[program:fastapi]
command=python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
directory=/app/backend
autostart=true
autorestart=true
priority=20
startretries=5
stdout_logfile=/var/log/supervisor/fastapi.log
stderr_logfile=/var/log/supervisor/fastapi_err.log
environment=PYTHONPATH="/app/backend",PYTHONUNBUFFERED="1"

; ── Streamlit dashboard ────────────────────────────────────────────────────────
[program:streamlit]
command=python3 -m streamlit run /app/streamlit_app/app.py --server.port=8502 --server.address=127.0.0.1 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --theme.base=dark --theme.backgroundColor="#010609" --theme.primaryColor="#00d2ff" --theme.textColor="#a8c8e0"
directory=/app/backend
autostart=true
autorestart=true
priority=30
startretries=5
stdout_logfile=/var/log/supervisor/streamlit.log
stderr_logfile=/var/log/supervisor/streamlit_err.log
environment=BACKEND_URL="http://localhost:8000/api",FRONTEND_URL="http://localhost:80",PYTHONPATH="/app/backend"
SUPEOF

# Port 8000 : FastAPI  — grader / REST API
# Port 80   : HTML frontend (Orbital Insight canvas)
# Port 8501 : Streamlit dashboard (nginx -> 8502)
EXPOSE 8000 80 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
