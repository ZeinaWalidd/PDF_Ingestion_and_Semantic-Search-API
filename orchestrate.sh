#!/usr/bin/env bash
#
# Usage: ./orchestrate.sh --action {start|terminate}

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

API_URL="http://localhost:8000"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-180}"

log()  { printf '[orchestrate] %s\n' "$*"; }
fail() { printf '[orchestrate] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    echo "Usage: ./orchestrate.sh --action {start|terminate}"
    echo
    echo "  start      Build images, start the app and vector DB, wait until healthy"
    echo "  terminate  Stop and remove containers, volumes and networks"
}

check_docker() {
    command -v docker >/dev/null 2>&1 \
        || fail "Docker is not installed or not on PATH."
    docker info >/dev/null 2>&1 \
        || fail "Docker is installed but the daemon is not running."
    docker compose version >/dev/null 2>&1 \
        || fail "Docker Compose v2 ('docker compose') is required."
}

start() {
    log "Building and starting containers (first build downloads the model, this can take a few minutes)..."
    if ! docker compose up --build --detach --remove-orphans \
            --wait --wait-timeout "$START_TIMEOUT_SECONDS"; then
        log "Services did not become healthy. Recent logs:"
        docker compose logs --tail 50 >&2 || true
        fail "Start failed. Run './orchestrate.sh --action terminate' to clean up."
    fi

    log "All services are healthy."
    log "API:     $API_URL"
    log "Docs:    $API_URL/docs"
    log "Logs:    docker compose logs -f app"
}

terminate() {
    log "Stopping and removing containers, volumes and networks..."
    docker compose down --volumes --remove-orphans
    log "Done."
}

ACTION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            [[ $# -ge 2 ]] || { usage; fail "--action needs a value."; }
            ACTION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            fail "Unknown argument: $1"
            ;;
    esac
done

case "$ACTION" in
    start)     check_docker; start ;;
    terminate) check_docker; terminate ;;
    "")        usage; exit 1 ;;
    *)         usage; fail "Unknown action: $ACTION" ;;
esac
