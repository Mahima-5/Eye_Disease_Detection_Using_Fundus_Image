#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   VisionaryAI Desktop — Setup        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Node
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found. Install from https://nodejs.org"
    exit 1
fi
echo "[OK] Node.js $(node -v)"

# Check Python
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python not found. Install from https://python.org"
    exit 1
fi
echo "[OK] Python: $($PYTHON_CMD --version)"

echo ""
echo "[1/3] Installing Python backend dependencies..."
cd backend
$PYTHON_CMD -m pip install -r requirements.txt --quiet
cd ..
echo "[OK] Python deps done"

echo ""
echo "[2/3] Installing Node.js root dependencies..."
npm install --quiet
echo "[OK] Node deps done"

echo ""
echo "[3/3] Setting up frontend..."
cd frontend
npm install --quiet 2>/dev/null || true
cd ..
echo "[OK] Frontend done"

echo ""
echo "══════════════════════════════════════"
echo "Setup complete!"
echo ""
echo "Start dev mode:   npm run dev"
echo "Build production: npm run build"
echo "══════════════════════════════════════"
echo ""

read -p "Start the app now? (y/n): " choice
if [ "$choice" = "y" ]; then
    npm run dev
fi
