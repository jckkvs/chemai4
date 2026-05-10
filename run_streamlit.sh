#!/usr/bin/env bash
# Streamlit版 ChemAI ML Studio を起動
[ -f .venv/bin/activate ] && source .venv/bin/activate
echo "ChemAI ML Studio (Streamlit) を起動しています..."
echo "  → http://localhost:8501"
cd frontend_streamlit
streamlit run app.py
