# ChemAI ML Studio — NiceGUI Edition

Pure Python 製フロントエンド。  
Streamlit / Django と同じ `backend/` を共有し、NiceGUI ベースのリッチ UI を提供します。

---

## 🚀 クイックスタート（最短 3 コマンド）

```bash
# 1. 依存パッケージのインストール
pip install -r requirements.txt

# 2. NiceGUI 版を起動
python frontend_nicegui/main.py

# 3. ブラウザで開く
#    → http://localhost:8080
```

> [!TIP]
> NiceGUI は自動でブラウザを開きます。開かない場合は手動で http://localhost:8080 にアクセスしてください。

---

## 📂 ディレクトリ構成

```
frontend_nicegui/
├── main.py                  # エントリーポイント（ポート 8080）
└── components/
    ├── __init__.py
    ├── data_tab.py          # 解析設定タブ（アップロード・列設定・SMILES・EDA・交差検証・推定機選択）
    ├── results_tab.py       # 結果確認タブ（モデル比較・評価・SHAP）
    ├── analysis_runner.py   # ワンクリック解析エンジン
    └── auto_params_ui.py    # パラメータ自動UI生成
```

---

## 👤 エンドユーザー向けガイド

### 初心者（最短 2 クリック）

1. **📂 データ読込**: CSV / Excel をドラッグ＆ドロップ  
   → 目的変数・SMILES 列を自動検出
2. **🚀 解析開始**: ヘッダーの「解析開始」ボタンを押す  
   → EDA → 前処理 → AutoML → 評価 → SHAP まで自動実行
3. **📊 結果確認**: 自動で結果タブに切り替わります

### 上級者（詳細設定）

| 設定項目 | 説明 |
|---------|------|
| **🏷️ 列の役割** | 目的変数・グループ列・時系列列の手動変更 |
| **⚗️ SMILES 特徴量** | 14 エンジンの記述子を個別 ON/OFF |
| **⚙️ パイプライン** | CV 分割数・使用モデル・スケーラー・単調性制約 |
| **📊 EDA** | データ品質チェック・統計量サマリー |

### サイドバー

左サイドバーでは進捗状況をリアルタイム表示します:

- ✅ データ読込 完了
- ✅ 目的変数設定 完了  
- ⬜ SMILES 検出 未検出
- ⬜ 解析完了 未完了

---

## 🔧 導入者（デプロイ担当）向けガイド

### 必須要件

| 項目 | 要件 |
|------|------|
| Python | 3.10 以上 |
| パッケージ | `requirements.txt` に記載 |
| NiceGUI | `pip install nicegui` （requirements.txt に含む） |

### インストール手順

```bash
# 1. リポジトリのクローン
git clone https://github.com/jckkvs/chemai2.git
cd chemai2

# 2. 仮想環境の作成（推奨）
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. 依存関係のインストール
pip install -r requirements.txt

# 4. オプション（量子化学計算を使う場合）
pip install mordred
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 起動オプション

```python
# frontend_nicegui/main.py の設定箇所
ui.run(
    title="ChemAI ML Studio",
    dark=True,          # ダークモード（True/False）
    port=8080,          # ポート番号
    reload=True,        # ホットリロード（開発時True）
    storage_secret="chemai-nicegui-secret",  # セッション暗号鍵
)
```

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `port` | 8080 | リッスンポート |
| `dark` | True | ダークモード有効 |
| `reload` | True | ファイル変更時の自動リロード（本番では False 推奨） |
| `host` | `0.0.0.0` | バインドアドレス（LAN 公開時） |

### 本番デプロイ

```bash
# ホットリロード無効 + 外部公開
python -c "
import sys; sys.path.insert(0, '.')
from frontend_nicegui.main import *
from nicegui import ui
ui.run(title='ChemAI ML Studio', dark=True, port=8080, reload=False, host='0.0.0.0')
"
```

> [!WARNING]
> 本番環境では `storage_secret` を安全なランダム文字列に変更してください。

### xTB バイナリ（量子化学記述子を使う場合）

```powershell
# Windows
$env:PATH = "$PWD\tools\xtb-6.7.1\bin;$env:PATH"
xtb --version
```

---

## 🖥️ 3 つのフロントエンド比較

| 版 | コマンド | ポート | 特徴 |
|---|---------|-------|------|
| **NiceGUI** ⭐ | `python frontend_nicegui/main.py` | **8080** | Pure Python UI、高速開発 |
| Streamlit | `streamlit run frontend_streamlit/app.py` | 8501 | データ分析向け、豊富なウィジェット |
| Django | `python frontend_django/manage.py runserver` | 8000 | REST API、ユーザー認証 |

---

## 🧪 テスト

```bash
# 全テスト
python -m pytest tests/ -v

# カバレッジ付き
python -m pytest tests/ --cov=backend --cov-branch --cov-report=term-missing
```

---

## 🔗 関連ドキュメント

- [プロジェクト README](../README.md) — 全体概要
- [REPRODUCE.md](../REPRODUCE.md) — 完全再現手順
- [backend/README.md](../backend/README.md) — バックエンド API ドキュメント
