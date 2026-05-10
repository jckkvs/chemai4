@echo off
chcp 65001 > nul
REM NiceGUI版 ChemAI ML Studio を起動
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
echo ChemAI ML Studio (NiceGUI) を起動しています...
echo   → http://localhost:8085
echo.
python frontend_nicegui/main.py
pause
