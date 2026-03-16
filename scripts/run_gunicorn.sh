#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Kill any existing instance
pkill -f "gunicorn.*app:app" 2>/dev/null; sleep 1

echo "Starting Genos API with Gunicorn on http://127.0.0.1:5000"
gunicorn -c gunicorn.conf.py app:app
