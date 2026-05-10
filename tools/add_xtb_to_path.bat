@echo off
REM ============================================================
REM add_xtb_to_path.bat — XTB を現在のセッション + 恒久的に PATH へ追加
REM ============================================================

set SCRIPT_DIR=%~dp0
set XTB_BIN=%SCRIPT_DIR%xtb-6.7.1\bin

if not exist "%XTB_BIN%\xtb.exe" (
    echo [エラー] xtb.exe が見つかりません: %XTB_BIN%
    echo.
    echo 以下から Windows 版バイナリをダウンロードしてください:
    echo   https://github.com/grimme-lab/xtb/releases
    echo   ファイル名: xtb-6.7.1-windows-x86_64.zip
    echo   解凍先: tools\xtb-6.7.1\
    pause
    exit /b 1
)

echo XTB バイナリを検出: %XTB_BIN%

REM 現在のセッション用
set PATH=%XTB_BIN%;%PATH%
echo 現在のセッションに XTB を追加しました。

REM 恒久的（ユーザー環境変数）
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set CURRENT_PATH=%%B
echo %CURRENT_PATH% | findstr /i "%XTB_BIN%" > nul
if errorlevel 1 (
    setx PATH "%CURRENT_PATH%;%XTB_BIN%"
    echo ユーザー環境変数 PATH に恒久的に追加しました。新しいターミナルで有効になります。
) else (
    echo すでに PATH に含まれています。
)

echo.
xtb --version
echo XTB の設定が完了しました。
pause
