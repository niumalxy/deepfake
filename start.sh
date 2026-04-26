#!/bin/bash

# ============================================
# Deepfake Agent Demo - Startup Script
# ============================================
# This script starts both the FastAPI main server
# and the Reflection Agent consumer concurrently.
# ============================================



set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uv sync

echo "============================================"
echo "  Deepfake Agent Demo - Starting Services"
echo "============================================"

# Cleanup function to kill all child processes on exit
cleanup() {
    echo ""
    echo "[INFO] Shutting down all services..."
    # Kill all child processes in the process group
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[INFO] All services stopped."
    exit 0
}

# Trap signals to ensure clean shutdown
trap cleanup SIGINT SIGTERM EXIT

# Create necessary directories
echo "[INFO] Checking necessary directories..."
mkdir -p agent/plan/docs
mkdir -p agent/summary/docs
mkdir -p segment_agent/plan/docs
mkdir -p segment_agent/summary/docs
echo "[INFO] Directories checked."

# ---- Start Reflection Agent ----
echo "[INFO] Starting Reflection Agent (RabbitMQ consumer)..."
uv run python -m reflection_agent.main &
REFLECTION_PID=$!
echo "[INFO] Reflection Agent started (PID: $REFLECTION_PID)"

# Give reflection agent a moment to initialize
sleep 2

# ---- Start FastAPI Main Server ----
echo "[INFO] Starting FastAPI Main Server on http://localhost:8000 ..."
uv run uvicorn main:app --host localhost --port 8000 &
MAIN_PID=$!
echo "[INFO] FastAPI Main Server started (PID: $MAIN_PID)"

echo ""
echo "============================================"
echo "  All services are running!"
echo "  - FastAPI Server:     http://localhost:8000"
echo "  - Reflection Agent:   Running (RabbitMQ)"
echo "============================================"
echo "  Press Ctrl+C to stop all services."
echo "============================================"

# Wait for any child process to exit
wait
