#!/usr/bin/env bash
# ============================================================
# setup.sh — ChemAI ML Studio ワンクリックセットアップ (Linux/macOS)
#
# 新規ユーザー向け:
#   chmod +x setup.sh && ./setup.sh
# ============================================================
set -e

echo ""
echo "========================================================"
echo "  ChemAI ML Studio - セットアップ"
echo "========================================================"
echo ""

# ── Python確認 ──
if ! command -v python3 &> /dev/null; then
    echo "[エラー] Python3 がインストールされていません。"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  macOS: brew install python"
    exit 1
fi
echo "  検出: $(python3 --version)"
echo ""

# ── 仮想環境作成 ──
if [ ! -d ".venv" ]; then
    echo "[1/5] 仮想環境を作成しています..."
    python3 -m venv .venv
    echo "       .venv を作成しました。"
else
    echo "[1/5] 仮想環境は既に存在します (.venv)"
fi
echo ""

# ── 仮想環境を有効化 ──
source .venv/bin/activate

# ── pip アップグレード ──
pip install --upgrade pip

# ── 必須パッケージ ──
echo "[2/5] 必須パッケージをインストールしています..."
pip install -r requirements.txt
echo ""

# ── オプショナルパッケージ ──
echo "[3/5] オプショナルパッケージをインストールしています..."
echo "       ※ 一部失敗してもアプリは動作します"
echo ""

install_optional() {
    local pkg="$1"
    local desc="$2"
    echo "--- $desc ---"
    pip install $pkg 2>/dev/null || echo "  [スキップ] $desc のインストールに失敗しました（アプリ動作に影響なし）"
}

# 化学記述子エンジン
install_optional "mordred" "Mordred (1800+ QSAR記述子)"
install_optional "unipka" "UniPKa (pKa/LogD予測)"
install_optional "ase" "ASE (量子化学インターフェース)"
install_optional "git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git" "openCOSMO-RS (溶媒和計算)"

# 深層学習系
install_optional "torch --index-url https://download.pytorch.org/whl/cpu" "PyTorch CPU版 (MolAI用)"
install_optional "scikit-fingerprints" "scikit-fingerprints (30+ フィンガープリント)"
install_optional "mol2vec" "Mol2Vec (Word2Vec分子埋め込み)"
install_optional "molfeat" "Molfeat (統合FPフレームワーク)"
install_optional "chemprop" "Chemprop (D-MPNN GNN)"
install_optional "padelpy" "PaDEL-Descriptor"
install_optional "descriptastorus" "DescriptaStorus (Merck高速記述子)"
install_optional "fairchem-core" "UMA / fairchem-core"

# 解釈性ツール
install_optional "sage-importance" "SAGE (ゲーム理論的重要度)"
install_optional "shapiq" "shapiq (Shapley Interaction)"
install_optional "imodels" "imodels (解釈可能モデル)"
install_optional "lime" "LIME (局所解釈)"
install_optional "eli5" "ELI5 (特徴量重要度)"

# パイプライン拡張
install_optional "mlxtend" "mlxtend"
install_optional "linear-tree" "linear-tree"
install_optional "group-lasso" "group-lasso"
echo ""

# ── xTB (Linux/macOS) ──
echo "[4/5] xTB バイナリの確認..."
if command -v xtb &> /dev/null; then
    echo "  xTB: 既にインストール済み ($(xtb --version 2>&1 | head -1))"
elif [ -f "tools/xtb-6.7.1/bin/xtb" ]; then
    echo "  xTB: tools/xtb-6.7.1/bin/xtb を検出"
    echo "  以下をシェル設定ファイルに追加してください:"
    echo "    export PATH=\"\$PATH:$(pwd)/tools/xtb-6.7.1/bin\""
else
    echo "  [情報] xTB が見つかりません。量子化学記述子を使う場合は:"
    echo "    conda install -c conda-forge xtb"
    echo "    または https://github.com/grimme-lab/xtb/releases からダウンロード"
fi
echo ""

# ── 動作確認 ──
echo "[5/5] 動作確認..."
echo ""
echo "--- 必須パッケージ ---"
python3 -c "import rdkit; print('  RDKit:', rdkit.__version__)" 2>/dev/null || echo "  RDKit: 未インストール [必須]"
python3 -c "import sklearn; print('  scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "  scikit-learn: 未インストール [必須]"
python3 -c "import nicegui; print('  NiceGUI: OK')" 2>/dev/null || echo "  NiceGUI: 未インストール [必須]"
echo ""
echo "--- 化学記述子エンジン (オプション) ---"
python3 -c "import mordred; print('  Mordred: OK')" 2>/dev/null || echo "  Mordred: 未インストール"
python3 -c "import unipka; print('  UniPKa: OK')" 2>/dev/null || echo "  UniPKa: 未インストール"
python3 -c "import torch; print('  PyTorch:', torch.__version__)" 2>/dev/null || echo "  PyTorch: 未インストール"
echo ""

echo "========================================================"
echo "  セットアップ完了！"
echo "========================================================"
echo ""
echo "  起動方法:"
echo "    source .venv/bin/activate"
echo "    python3 frontend_nicegui/main.py"
echo "    → http://localhost:8085"
echo ""
