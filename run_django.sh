#!/usr/bin/env bash
# Django版 ChemAI ML Studio を起動
[ -f .venv/bin/activate ] && source .venv/bin/activate
echo "ChemAI ML Studio (Django) を起動しています..."
echo "  → http://localhost:8000"
python3 frontend_django/manage.py runserver 0.0.0.0:8000
