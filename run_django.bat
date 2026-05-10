@echo off
chcp 65001 > nul
REM Django版 ChemAI ML Studio を起動
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
echo ChemAI ML Studio (Django) を起動しています...
echo   → http://localhost:8000
echo.
python frontend_django/manage.py runserver 0.0.0.0:8000
pause
