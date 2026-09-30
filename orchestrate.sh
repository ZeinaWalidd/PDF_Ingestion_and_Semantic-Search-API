#!/usr/bin/env bash

set -e

ACTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            ACTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

case "$ACTION" in
    start)
        echo "Starting application..."
        docker compose up --build -d
        ;;

    terminate)
        echo "Stopping application..."
        docker compose down -v
        ;;

    *)
        echo "Usage: ./orchestrate.sh --action {start|terminate}"
        exit 1
        ;;
esac