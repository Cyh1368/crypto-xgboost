#!/bin/bash

# ==============================================================================
# Script Name: host_paper_trading.sh
# Description: Sets up a brand new Ubuntu machine to host the paper trading script.
#              - Updates system packages
#              - Installs Python 3, venv, and git
#              - Installs Node.js and PM2
#              - Clones the repository
#              - Sets up a virtual environment and installs Python dependencies
#              - Starts the script using PM2
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
REPO_URL="https://github.com/Cyh1368/crypto-xgboost.git"
REPO_DIR="crypto-xgboost"
VENV_NAME="paper-trading-venv"
SCRIPT_PATH="backtester_v1/paper_trading/run_paper_trading.py"

echo ">>> Starting system update and dependency installation..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl

# --- Node.js & PM2 Setup ---
if ! command -v node &> /dev/null; then
    echo ">>> Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

if ! command -v pm2 &> /dev/null; then
    echo ">>> Installing PM2..."
    sudo npm install -g pm2
fi

# --- Repository Setup ---
if [ -d "$REPO_DIR" ]; then
    echo ">>> Directory $REPO_DIR already exists. Updating..."
    cd "$REPO_DIR"
    git pull
else
    echo ">>> Cloning repository: $REPO_URL"
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi

# --- Virtual Environment Setup ---
if [ ! -d "$VENV_NAME" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv "$VENV_NAME"
fi

echo ">>> Installing Python dependencies..."
source "$VENV_NAME"/bin/activate
pip install --upgrade pip
# Explicitly installing scikit-learn and scipy as they are often required by joblib loaders
pip install pandas numpy joblib xgboost ccxt scikit-learn scipy

# --- Start the Script with PM2 ---
echo ">>> Starting the paper trading script with PM2..."
# We use the absolute path to the venv python to ensure it uses the correct environment
VENV_PYTHON="$(pwd)/$VENV_NAME/bin/python"

# Stop existing process if it exists to avoid duplicates on re-run
pm2 stop paper-trading &>/dev/null || true
pm2 delete paper-trading &>/dev/null || true

# Start the script
# We set the working directory to the repo root so imports work
pm2 start "$VENV_PYTHON" --name "paper-trading" -- "$SCRIPT_PATH"

# Save PM2 process list to resurrect on reboot
pm2 save

echo "=============================================================================="
echo "Setup Complete!"
echo "The paper trading script is now running in the background via PM2."
echo ""
echo "Commands to monitor:"
echo "  pm2 status          - Check if the script is running"
echo "  pm2 logs paper-trading - View live logs"
echo "  pm2 stop paper-trading - Stop the script"
echo "  pm2 start paper-trading - Start the script"
echo "=============================================================================="
