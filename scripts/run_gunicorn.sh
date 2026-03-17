#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

BIND_ADDR="${GENOS_API_BIND:-127.0.0.1:6000}"
PORT="${BIND_ADDR##*:}"

# Kill all gunicorn processes and anything holding the port
pkill -9 -f "gunicorn" 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true

# Wait until port is actually free
for i in {1..10}; do
    fuser "${PORT}/tcp" 2>/dev/null || break
    sleep 1
done

echo "Starting Genos API with Gunicorn on http://${BIND_ADDR}"
gunicorn -c gunicorn.conf.py app:app
