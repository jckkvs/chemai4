@echo off
chcp 65001 > nul
REM Streamlit版 ChemAI ML Studio を起動
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
echo ChemAI ML Studio (Streamlit) を起動しています...
echo   → http://localhost:8501
echo.
cd frontend_streamlit
streamlit run app.py
pause
