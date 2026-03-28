#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

DEFAULT_BIND="${GENOS_API_BIND:-127.0.0.1:6001}"
HEALTH_TIMEOUT=120
HEALTH_RETRY_INTERVAL=2

log_info() {
    echo "[*] $*"
}

log_ok() {
    echo "[+] $*"
}

log_err() {
    echo "[-] $*" >&2
}

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN
    else
        lsof -ti "tcp:${port}" -sTCP:LISTEN >/dev/null 2>&1
    fi
}

# Check if API is healthy on a given bind address
health_check() {
    local bind="$1"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://${bind}/health" 2>/dev/null || echo "000")
    [ "$code" = "200" ]
}

# Wait for API health endpoint with timeout
wait_healthy() {
    local bind="$1"
    local elapsed=0

    log_info "Waiting for API to be healthy..."
    while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
        if health_check "$bind"; then
            log_ok "API is healthy on ${bind} (${elapsed}s)"
            return 0
        fi
        sleep "$HEALTH_RETRY_INTERVAL"
        elapsed=$((elapsed + HEALTH_RETRY_INTERVAL))
    done

    log_err "API health check timed out after ${HEALTH_TIMEOUT}s"
    return 1
}

# Kill any running gunicorn processes
stop_gunicorn() {
    local bind="$1"
    local port="${bind##*:}"
    local pids

    log_info "Checking for existing process on ${bind}..."
    if ! port_in_use "$port"; then
        log_info "No API listener found on port ${port}."
        return 0
    fi

    log_info "Port ${port} is in use. Stopping existing process(es)..."
    pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null | sort -u | tr '\n' ' ' || true)"
    if [ -n "${pids// /}" ]; then
        # shellcheck disable=SC2086
        kill ${pids} 2>/dev/null || true
    fi

    # Fallback for listeners not visible/killable by current user.
    pkill -f "gunicorn.*127.0.0.1:${port}" 2>/dev/null || true
    pkill -f "gunicorn.*:${port}" 2>/dev/null || true

    if port_in_use "$port" && command -v sudo >/dev/null 2>&1; then
        sudo -n fuser -k -n tcp "$port" >/dev/null 2>&1 || true
    fi

    for _ in $(seq 1 40); do
        if ! port_in_use "$port"; then
            return 0
        fi
        sleep 0.25
    done

    if port_in_use "$port"; then
        log_err "Port ${port} is still in use. Could not stop existing process."
        log_err "Re-run with sudo or stop the existing service manually."
        return 1
    fi
}

# Activate Python virtual environment
activate_venv() {
    if [ ! -f "venv/bin/activate" ]; then
        log_err "venv/bin/activate not found"
        return 1
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
}

# Start gunicorn on the specified bind address
start_gunicorn() {
    local bind="$1"
    log_info "Starting gunicorn on ${bind}..."
    GENOS_API_BIND="$bind" gunicorn -c gunicorn.conf.py app:app &
    local pid=$!
    echo "$pid"
}

# Main reload command
cmd_reload() {
    local bind="$DEFAULT_BIND"
    
    log_info "Reloading API on ${bind}..."

    if [ "$bind" != "127.0.0.1:6001" ]; then
        log_err "This script is restricted to internal API bind 127.0.0.1:6001 (got: ${bind})"
        return 1
    fi
    
    stop_gunicorn "$bind" || return 1
    activate_venv || return 1
    
    local retry=0
    local max_retries=3
    
    while [ $retry -lt $max_retries ]; do
        log_info "Starting gunicorn (attempt $((retry + 1))/$max_retries)..."
        
        start_gunicorn "$bind"
        
        if wait_healthy "$bind"; then
            log_ok "Internal API reload complete and verified on ${bind}"
            return 0
        fi
        
        log_err "Health check failed, retrying..."
        stop_gunicorn "$bind" || return 1
        retry=$((retry + 1))
    done
    
    log_err "Failed to start API after $max_retries attempts"
    return 1
}

# Check status
cmd_status() {
    log_info "Checking API status..."
    
    if health_check "$DEFAULT_BIND"; then
        log_ok "API is healthy on ${DEFAULT_BIND}"
        return 0
    else
        log_err "API is not healthy on ${DEFAULT_BIND}"
        return 1
    fi
}

# Help text
usage() {
    cat <<EOF
Usage: $(basename "$0") [reload|status]

    reload  Restart internal API gunicorn on 127.0.0.1:6001 (default)
  status  Check if API is healthy
  -h      Show this help message
EOF
}

# Main entry point
MODE="${1:-reload}"
case "$MODE" in
    reload)
        cmd_reload
        ;;
    status)
        cmd_status
        ;;
    -h|--help)
        usage
        ;;
    *)
        log_err "Unknown mode: $MODE"
        usage
        exit 1
        ;;
esac
