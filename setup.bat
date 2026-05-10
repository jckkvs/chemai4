@echo off
chcp 65001 > nul
REM ============================================================
REM setup.bat — ChemAI ML Studio ワンクリックセットアップ (Windows)
REM
REM 新規ユーザー向け: ダブルクリックで完全自動セットアップ
REM   1. 仮想環境の作成（なければ）
REM   2. 必須パッケージインストール
REM   3. オプショナルパッケージインストール（失敗してもOK）
REM   4. xTB PATH設定
REM   5. 動作確認
REM ============================================================

echo.
echo ========================================================
echo   ChemAI ML Studio - セットアップ
echo ========================================================
echo.

REM ── Python確認 ──
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [エラー] Python がインストールされていません。
    echo   https://www.python.org/downloads/ からインストールしてください。
    echo   ※ インストール時に「Add Python to PATH」にチェックを入れてください。
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   検出: %%v
echo.

REM ── 仮想環境作成 ──
if not exist .venv (
    echo [1/5] 仮想環境を作成しています...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [エラー] 仮想環境の作成に失敗しました。
        pause
        exit /b 1
    )
    echo       .venv を作成しました。
) else (
    echo [1/5] 仮想環境は既に存在します (.venv)
)
echo.

REM ── 仮想環境を有効化 ──
call .venv\Scripts\activate.bat

REM ── 以降は tools/setup_all.bat に委譲 ──
echo [2/5] tools\setup_all.bat を実行します...
echo       ※ オプションパッケージが一部失敗してもアプリは動作します
echo.
call tools\setup_all.bat

echo.
echo ========================================================
echo   セットアップ完了！
echo ========================================================
echo.
echo   起動方法:
echo     NiceGUI版:   run_nicegui.bat をダブルクリック
echo     Streamlit版: run_streamlit.bat をダブルクリック
echo     Django版:    run_django.bat をダブルクリック
echo.
echo   または PowerShell で:
echo     .\.venv\Scripts\Activate.ps1
echo     python frontend_nicegui/main.py
echo.
pause
