#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required to install Python 3.11 on this machine."
  exit 1
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Installing python@3.11 with Homebrew..."
  brew install python@3.11
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  python3.11 -m venv "${VENV_DIR}"
fi

echo "Upgrading pip tooling..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing Frame-Scope dependencies..."
"${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}"

echo
echo "Bootstrap complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Check auth with: ${VENV_DIR}/bin/hf auth whoami"
