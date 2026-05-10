#!/usr/bin/env bash
# NiceGUI版 ChemAI ML Studio を起動
[ -f .venv/bin/activate ] && source .venv/bin/activate
echo "ChemAI ML Studio (NiceGUI) を起動しています..."
echo "  → http://localhost:8085"
python3 frontend_nicegui/main.py
