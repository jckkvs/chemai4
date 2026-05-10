@echo off
chcp 65001 > nul
REM ============================================================
REM setup_all.bat — chemai2 全自動セットアップスクリプト
REM 初回セットアップ or 新PC移行時に実行してください
REM
REM ※ オプショナルパッケージのインストールが一部失敗しても
REM   アプリは正常に起動します（未対応エンジンはグレーアウト）
REM ============================================================

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║   chemai2 全自動セットアップ                             ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM スクリプトのディレクトリ = tools/
set TOOLS_DIR=%~dp0
set ROOT_DIR=%TOOLS_DIR%..
set XTB_BIN=%TOOLS_DIR%xtb-6.7.1\bin

echo ============================================================
echo [1/7] 必須パッケージをインストールしています...
echo ============================================================
echo.
pip install -r "%ROOT_DIR%\requirements.txt"

echo.
echo ============================================================
echo [2/7] 化学記述子エンジン（オプショナル）
echo       ※ 失敗してもアプリは動作します
echo ============================================================
echo.
echo --- Mordred (1800+ QSAR記述子) ---
pip install mordred
echo --- UniPKa (pKa/LogD予測) ---
pip install unipka
echo --- ASE (量子化学インターフェース) ---
pip install ase
echo --- openCOSMO-RS (溶媒和計算) ---
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git

echo.
echo ============================================================
echo [3/7] 深層学習系エンジン（オプショナル）
echo       ※ PyTorch CPU版をインストールします
echo ============================================================
echo.
echo --- PyTorch CPU版 (MolAI用) ---
pip install torch --index-url https://download.pytorch.org/whl/cpu
echo --- scikit-fingerprints (30+ フィンガープリント) ---
pip install scikit-fingerprints
echo --- Mol2Vec (Word2Vec分子埋め込み) ---
pip install mol2vec
echo --- Molfeat (統合FPフレームワーク) ---
pip install molfeat
echo --- Chemprop (D-MPNN GNN) ---
pip install chemprop
echo --- PaDEL-Descriptor (Java必要) ---
pip install padelpy
echo --- DescriptaStorus (Merck高速記述子) ---
pip install descriptastorus
echo --- UMA / fairchem-core ---
pip install fairchem-core

echo.
echo ============================================================
echo [4/7] 解釈性・説明性ツール（オプショナル）
echo ============================================================
echo.
echo --- SAGE (ゲーム理論的特徴量重要度) ---
pip install sage-importance
echo --- shapiq (Shapley Interaction) ---
pip install shapiq
echo --- imodels (解釈可能モデル: FIGS, RuleFit等) ---
pip install imodels
echo --- LIME (局所解釈) ---
pip install lime
echo --- ELI5 (特徴量重要度) ---
pip install eli5

echo.
echo ============================================================
echo [5/7] パイプライン拡張（オプショナル）
echo ============================================================
echo.
echo --- mlxtend ---
pip install mlxtend
echo --- linear-tree ---
pip install linear-tree
echo --- group-lasso ---
pip install group-lasso

echo.
echo ============================================================
echo [6/7] XTB バイナリ PATH を設定しています...
echo ============================================================
echo      XTB バイナリ場所: %XTB_BIN%
if exist "%XTB_BIN%\xtb.exe" (
    echo      xtb.exe を検出しました。
    REM ユーザー環境変数に追加（恒久的）
    for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set CURRENT_PATH=%%B
    echo %CURRENT_PATH% | findstr /i "%XTB_BIN%" > nul
    if errorlevel 1 (
        setx PATH "%CURRENT_PATH%;%XTB_BIN%"
        echo      PATH に XTB を追加しました。新しいターミナルで有効になります。
    ) else (
        echo      XTB はすでに PATH に含まれています。
    )
) else (
    echo      [情報] xtb.exe が見つかりません: %XTB_BIN%
    echo      量子化学記述子を使いたい場合は以下からダウンロードしてください:
    echo        https://github.com/grimme-lab/xtb/releases
    echo        ファイル名: xtb-6.7.1-windows-x86_64.zip
    echo        解凍先: tools\xtb-6.7.1\
)

echo.
echo ============================================================
echo [7/7] 動作確認...
echo ============================================================
echo.
echo --- 必須パッケージ ---
python -c "import rdkit; print('  RDKit:', rdkit.__version__)" 2>nul || echo   RDKit: 未インストール [必須]
python -c "import sklearn; print('  scikit-learn:', sklearn.__version__)" 2>nul || echo   scikit-learn: 未インストール [必須]
python -c "import nicegui; print('  NiceGUI: OK')" 2>nul || echo   NiceGUI: 未インストール [必須]
echo.
echo --- 化学記述子エンジン（オプション）---
python -c "import mordred; print('  Mordred: OK')" 2>nul || echo   Mordred: 未インストール
python -c "import unipka; print('  UniPKa: OK')" 2>nul || echo   UniPKa: 未インストール
python -c "import ase; print('  ASE:', __import__('ase').__version__)" 2>nul || echo   ASE: 未インストール
python -c "import opencosmorspy; print('  openCOSMO-RS: OK')" 2>nul || echo   openCOSMO-RS: 未インストール
where xtb >nul 2>&1 && echo   XTB: OK || echo   XTB: PATH 未設定
echo.
echo --- 深層学習系（オプション）---
python -c "import torch; print('  PyTorch:', torch.__version__)" 2>nul || echo   PyTorch: 未インストール
python -c "import skfp; print('  scikit-fingerprints: OK')" 2>nul || echo   scikit-fingerprints: 未インストール
python -c "import mol2vec; print('  Mol2Vec: OK')" 2>nul || echo   Mol2Vec: 未インストール
python -c "import molfeat; print('  Molfeat: OK')" 2>nul || echo   Molfeat: 未インストール
python -c "import chemprop; print('  Chemprop: OK')" 2>nul || echo   Chemprop: 未インストール
echo.
echo --- 解釈性ツール（オプション）---
python -c "import sage; print('  SAGE: OK')" 2>nul || echo   SAGE: 未インストール
python -c "import shapiq; print('  shapiq:', shapiq.__version__)" 2>nul || echo   shapiq: 未インストール
python -c "import shap; print('  SHAP:', shap.__version__)" 2>nul || echo   SHAP: 未インストール
python -c "import imodels; print('  imodels: OK')" 2>nul || echo   imodels: 未インストール

echo.
echo ============================================================
echo セットアップ完了！以下でアプリを起動できます:
echo.
echo   python frontend_nicegui/main.py
echo   → http://localhost:8085
echo.
echo   ※ 一部パッケージが未インストールでもアプリは起動します。
echo      未対応のエンジンはUI上でグレーアウト表示されます。
echo ============================================================
echo.
pause
