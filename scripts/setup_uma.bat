@echo off
REM ============================================================
REM  UMA (Universal Model for Atoms) セットアップスクリプト
REM  Meta FAIR の分子特性予測モデルをセットアップします。
REM ============================================================

echo.
echo ========================================
echo   UMA セットアップ
echo   Meta FAIR - Universal Model for Atoms
echo ========================================
echo.

REM 1. fairchem-core のインストール
echo [1/3] fairchem-core をインストール中...
pip install fairchem-core
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] fairchem-core のインストールに失敗しました。
    echo   PyTorch が未インストールの場合は先にインストールしてください:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)
echo [OK] fairchem-core インストール完了
echo.

REM 2. HuggingFace 認証
echo [2/3] HuggingFace にログインしてください。
echo   ※ https://huggingface.co/facebook/UMA にアクセス申請済みであることを確認してください。
echo   ※ アクセストークンは https://huggingface.co/settings/tokens で取得できます。
echo.
huggingface-cli login
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] HuggingFace ログインに失敗しました。
    echo   手動でログインしてください: huggingface-cli login
)
echo.

REM 3. モデルの事前ダウンロード（初回実行を高速化）
echo [3/3] UMA モデルを事前ダウンロード中...
echo   ※ モデルサイズ: uma-s (Small) 約200MB
python -c "from fairchem.core import pretrained_mlip; p = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu'); print('[OK] UMA Small モデルのダウンロード完了')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] モデルのダウンロードに失敗しました。
    echo   HuggingFace にログイン済みか確認してください。
    echo   https://huggingface.co/facebook/UMA でアクセスが承認されているか確認してください。
    pause
    exit /b 1
)
echo.

echo ========================================
echo   セットアップ完了！
echo ========================================
echo.
echo 使い方:
echo   ChemAI ML Studio の「⚗️ SMILES特徴量設計」タブで
echo   「UMA」エンジンをONにしてください。
echo.
echo   利用可能モデル:
echo     uma-s-1p2  : Small (高速・CPU推奨)
echo     uma-m-1p1  : Medium (高精度・GPU推奨)
echo.
pause
