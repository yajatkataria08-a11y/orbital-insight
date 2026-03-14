FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev nginx curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Streamlit requirements
RUN pip3 install --no-cache-dir streamlit altair pandas requests

COPY backend/main.py .
COPY frontend/index.html /var/www/html/index.html
COPY streamlit_app/app.py /app/streamlit_app/app.py

# Nginx config: frontend on :80, API proxy, Streamlit proxy on :8501
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
    location /docs { proxy_pass http://127.0.0.1:8000/docs; }
    location /openapi.json { proxy_pass http://127.0.0.1:8000/openapi.json; }
}

server {
    listen 8501;
    server_name _;
    location / {
        proxy_pass         http://127.0.0.1:8502/;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }
}
NGINXEOF

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Port 8000: FastAPI backend (NSH grader required)
# Port 80:   Orbital Insight HTML frontend
# Port 8501: Streamlit analytical dashboard (via nginx)
EXPOSE 8000 80 8501

CMD ["/app/start.sh"]
