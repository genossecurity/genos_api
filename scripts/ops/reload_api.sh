#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

BIND="${GENOS_API_BIND:-127.0.0.1:6001}"
HOST="${BIND%%:*}"
PORT="${BIND##*:}"

echo "[*] Killing existing genos_api gunicorn..."
pkill -9 -f "genos_api.*gunicorn" 2>/dev/null || true
pkill -9 -f "gunicorn -c gunicorn.conf.py" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true

# Wait until port is free
for i in {1..10}; do
    fuser "${PORT}/tcp" 2>/dev/null || break
    sleep 1
done

echo "[*] Starting genos_api gunicorn on ${BIND}..."
source venv/bin/activate
gunicorn -c gunicorn.conf.py app:app &
GUNICORN_PID=$!

# Wait for health check (model loading + warm-up)
echo "[*] Waiting for engine warm-up..."
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://${BIND}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "[+] Genos API is live on ${BIND} (took ${ELAPSED}s)"
        exit 0
    fi
    # Make sure gunicorn is still running
    if ! kill -0 "$GUNICORN_PID" 2>/dev/null; then
        echo "[-] ERROR: gunicorn died during startup"
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

echo "[-] ERROR: health check timed out after ${MAX_WAIT}s"
kill -9 "$GUNICORN_PID" 2>/dev/null || true
exit 1
