#!/bin/bash
set -e
echo "=================================================="
echo "  ORBITAL INSIGHT v6.0 — NSH 2026 ACM"
echo "  Physics: RK4 + J2 | Hohmann Recovery | Chan Pc"
echo "  Blind CDM | Optimal ΔV | Proactive SK"
echo "=================================================="

service nginx start
echo "[OK] Nginx started (frontend :80 | streamlit proxy :8501)"

cd /app/backend
echo "[*] Starting FastAPI physics engine on 0.0.0.0:8000..."
python3 main.py &
BACKEND_PID=$!

sleep 3
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[OK] Backend running (PID $BACKEND_PID)"
else
    echo "[FAIL] Backend failed to start"
    exit 1
fi

echo "[*] Starting Streamlit dashboard on 0.0.0.0:8502..."
cd /app
streamlit run streamlit_app/app.py \
    --server.port 8502 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --theme.base dark \
    --theme.backgroundColor "#010609" \
    --theme.primaryColor "#00d2ff" \
    --theme.textColor "#a8c8e0" &
STREAMLIT_PID=$!

sleep 4
if kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "[OK] Streamlit running (PID $STREAMLIT_PID)"
else
    echo "[WARN] Streamlit may still be starting..."
fi

echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │  Orbital Insight:  http://localhost:80       │"
echo "  │  FastAPI Docs:     http://localhost:8000     │"
echo "  │  Streamlit:        http://localhost:8501     │"
echo "  │  API Direct:       http://localhost:8000/api │"
echo "  └─────────────────────────────────────────────┘"
echo "=================================================="

wait $BACKEND_PID
