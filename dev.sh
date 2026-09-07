#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/compose.yml"

cd "$ROOT_DIR"

usage() {
    cat <<EOF
Usage: ./dev.sh [command] [service]

Commands:
  up        Build and start all services in dev mode (default), logs attached
  upd       Same as up, but detached (runs in background)
  down      Stop and remove all services
  restart   Restart all services (or one: ./dev.sh restart api)
  logs      Tail logs (all services, or one: ./dev.sh logs llm-worker)
  ps        Show status of running services
  sh        Open a shell in a running service (./dev.sh sh api)

Services: nginx, frontend, api, llm-worker, db, rabbitmq, flower
EOF
}

check_env() {
    if [[ ! -f "$ROOT_DIR/.env" ]]; then
        echo "Missing $ROOT_DIR/.env (needed for db/broker/api credentials)." >&2
        exit 1
    fi
    if [[ ! -f "$ROOT_DIR/frontend/.env" ]]; then
        echo "Missing $ROOT_DIR/frontend/.env (needed for VITE_BASE_URL)." >&2
        exit 1
    fi
}

print_urls() {
    cat <<EOF
Services will be available through nginx at:
  Frontend      http://localhost/
  Backend API   http://localhost/backend/api/v1/
  Flower        http://localhost/flower/
  RabbitMQ UI   http://localhost/rabbitmq/

EOF
}

cmd="${1:-up}"
shift || true

case "$cmd" in
    up)
        check_env
        print_urls
        docker compose -f "$COMPOSE_FILE" up --build
        ;;
    upd)
        check_env
        print_urls
        docker compose -f "$COMPOSE_FILE" up --build -d
        ;;
    down)
        docker compose -f "$COMPOSE_FILE" down
        ;;
    restart)
        docker compose -f "$COMPOSE_FILE" restart "$@"
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f "$@"
        ;;
    ps)
        docker compose -f "$COMPOSE_FILE" ps
        ;;
    sh)
        if [[ -z "${1:-}" ]]; then
            echo "Usage: ./dev.sh sh <service>" >&2
            exit 1
        fi
        docker compose -f "$COMPOSE_FILE" exec "$1" sh
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        usage
        exit 1
        ;;
esac
