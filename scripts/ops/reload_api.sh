#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

DEFAULT_BIND="${GENOS_API_BIND:-127.0.0.1:6001}"
MAX_WAIT=120

usage() {
    cat <<'EOF'
Usage: scripts/ops/reload_api.sh [reload|start|nginx|status]

  reload  Restart gunicorn, wait for /health, reload nginx, verify API (default)
  start   Start gunicorn in foreground (replaces run_gunicorn.sh behavior)
  nginx   Reload nginx only and verify API health on active port
  status  Print active API port (6001/6000) and health status
EOF
}

detect_active_bind() {
    local preferred
    if preferred=$(detect_nginx_proxy_bind); then
        if curl -s -o /dev/null -w "%{http_code}" "http://${preferred}/health" | grep -q '^200$'; then
            echo "$preferred"
            return 0
        fi
    fi

    for port in 6001 6000; do
        if [ "127.0.0.1:${port}" = "${preferred:-}" ]; then
            continue
        fi
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" | grep -q '^200$'; then
            echo "127.0.0.1:${port}"
            return 0
        fi
    done
    return 1
}

detect_nginx_proxy_bind() {
    local conf="/etc/nginx/sites-enabled/genossec.com"
    if [ -r "$conf" ]; then
        # Parse first proxy_pass target like http://127.0.0.1:6000
        local target
        target=$(grep -Eo 'proxy_pass[[:space:]]+http://127\.0\.0\.1:[0-9]+' "$conf" | head -n1 | awk -F'http://' '{print $2}')
        if [ -n "${target:-}" ]; then
            echo "$target"
            return 0
        fi
    fi
    return 1
}

resolve_bind_for_reload() {
    if bind=$(detect_nginx_proxy_bind); then
        echo "$bind"
    elif bind=$(detect_active_bind); then
        echo "$bind"
    else
        echo "$DEFAULT_BIND"
    fi
}

activate_venv() {
    if [ ! -f "venv/bin/activate" ]; then
        echo "[-] ERROR: venv/bin/activate not found"
        exit 1
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
}

kill_existing_gunicorn() {
    local port="$1"
    echo "[*] Killing existing genos_api gunicorn..."
    pkill -9 -f "genos_api.*gunicorn" 2>/dev/null || true
    pkill -9 -f "gunicorn -c gunicorn.conf.py" 2>/dev/null || true
    fuser -k "${port}/tcp" 2>/dev/null || true

    for _ in {1..10}; do
        fuser "${port}/tcp" 2>/dev/null || break
        sleep 1
    done
}

wait_for_health() {
    local bind="$1"
    local elapsed=0

    echo "[*] Waiting for engine warm-up..."
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        status=$(curl -s -o /dev/null -w "%{http_code}" "http://${bind}/health" 2>/dev/null || echo "000")
        if [ "$status" = "200" ]; then
            echo "[+] Genos API is live on ${bind} (took ${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "[-] ERROR: health check timed out after ${MAX_WAIT}s"
    return 1
}

reload_nginx() {
    echo "[*] Reloading nginx..."
    sudo nginx -s reload
    echo "[+] Nginx reloaded successfully"
    sleep 2
}

verify_api() {
    local bind="$1"
    echo "[*] Testing API endpoint at http://${bind}/health..."
    health_status=$(curl -s -o /dev/null -w "%{http_code}" "http://${bind}/health" 2>/dev/null || echo "000")
    if [ "$health_status" = "200" ]; then
        echo "[+] API health check passed (HTTP 200)"
        return 0
    fi
    echo "[-] ERROR: API health check failed (HTTP ${health_status})"
    return 1
}

cmd_reload() {
    bind="$(resolve_bind_for_reload)"
    port="${bind##*:}"

    echo "[*] Reloading API on ${bind}..."
    kill_existing_gunicorn "$port"
    activate_venv

    echo "[*] Starting genos_api gunicorn on ${bind}..."
    GENOS_API_BIND="$bind" gunicorn -c gunicorn.conf.py app:app &
    gunicorn_pid=$!

    if ! wait_for_health "$bind"; then
        kill -9 "$gunicorn_pid" 2>/dev/null || true
        exit 1
    fi

    reload_nginx
    verify_api "$bind"
    echo "[+] Reload complete and verified"
}

cmd_start() {
    bind="$(resolve_bind_for_reload)"
    port="${bind##*:}"
    echo "[*] Starting API on ${bind}..."
    kill_existing_gunicorn "$port"
    activate_venv
    exec GENOS_API_BIND="$bind" gunicorn -c gunicorn.conf.py app:app
}

cmd_nginx() {
    if ! bind=$(detect_active_bind); then
        echo "[-] ERROR: No API found running on port 6001 or 6000"
        exit 1
    fi
    echo "[*] Found active API on ${bind}"
    reload_nginx
    verify_api "$bind"
    echo "[+] Nginx reload complete and verified"
}

cmd_status() {
    if bind=$(detect_nginx_proxy_bind); then
        echo "[*] Nginx proxy target: ${bind}"
    else
        echo "[*] Nginx proxy target: unknown"
    fi

    if bind=$(detect_active_bind); then
        echo "[+] API healthy on ${bind}"
    else
        echo "[-] API not healthy on ports 6001 or 6000"
        exit 1
    fi
}

MODE="${1:-reload}"
case "$MODE" in
    reload)
        cmd_reload
        ;;
    start)
        cmd_start
        ;;
    nginx)
        cmd_nginx
        ;;
    status)
        cmd_status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[-] ERROR: Unknown mode '$MODE'"
        usage
        exit 1
        ;;
esac
