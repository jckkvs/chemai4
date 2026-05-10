# ChemAI ML Studio (chemai2)

化学構造データ（SMILES）とテーブルデータを統合して、探索的データ解析 → 機械学習モデル構築 → モデル解釈までを一気通貫で行う **ケモインフォマティクス統合プラットフォーム** です。

量子化学計算（GFN2-xTB）、pKa予測（UniPKa）、溶媒和計算（COSMO-RS）による高精度記述子と、
直感的な NiceGUI UI でワンクリック AutoML を実現します。

---

## ✨ 主な機能

### 🧪 ケモインフォマティクス

- **SMILES → 自動記述子生成**（7種の計算エンジン搭載）
  - **RDKit**: 物理化学記述子（MolWt, LogP, TPSA 等）＋ Morgan/RDKit/MACCS フィンガープリント
  - **Mordred**: 1,800+ QSAR 記述子（2Dトポロジカル）
  - **GFN2-xTB**: 半経験的量子化学記述子 — HOMO/LUMO/双極子モーメント/Mulliken電荷
  - **UniPKa**: pKa (酸性/塩基性) / LogD / 溶媒和エネルギー
  - **COSMO-RS**: σ-プロファイルによる溶媒和自由エネルギー
  - **MolAI**: CNN + PCA 分子潜在空間ベクトル
  - **GroupContrib**: 基団寄与法による熱物性推定

- **⚡ 分子電荷・スピン設定UI**
  - 形式電荷（SMILES自動読取 or 手動指定）・スピン多重度の設定
  - pH依存プロトン化（UniPKa pKa + Henderson-Hasselbalch式）
  - 5モード: as_is / neutral / auto_ph / max_acid / max_base
  - Gasteiger部分電荷の色付きSVG可視化
  - 分子ごとの個別電荷設定（上級者向け）
  - XTB に `--chrg` / `--uhf` を自動連携

- **高度な記述子選択UI**（5つの視点）
  - 相関係数、数え上げ変数、目的変数系統、物理的意味、計算ライブラリ

### 🤖 機械学習

- **AutoML**（ワンクリック）: Random Forest, LightGBM, XGBoost, CatBoost, SVR, KNN 等
- **Optuna 自動チューニング** + Grid Search
- **全組み合わせ Pipeline 探索**: 前処理・特徴選択・推定器の直積をCV評価
- **高度なモデル群**:
  - LinearTree / LinearForest / LinearBoost（区分線形モデル）
  - RGF (Regularized Greedy Forest)
  - imodels（解釈可能モデル）
  - 単調制約付きカーネルモデル
- **交差検証**: KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit 等

### 📊 データ分析・前処理

- EDA（探索的データ解析）・相関分析・外れ値検出
- 自動欠損値補完・カテゴリカルエンコーディング
- PCA, t-SNE, UMAP, PHATE 次元削減・可視化

### 🔬 モデル解釈 (XAI)

- **SHAP**: Summary / Individual / Dependence / Heatmap / Interaction
- **SAGE**: 特徴量重要度のゲーム理論的評価
- **shapiq**: SRI 分解による特徴量相互作用
- 伝統的手法: Feature Importance / Permutation Importance / PDP

---

## 📂 ディレクトリ構成

```
chemai2/
├── backend/                    # 解析・学習コアロジック
│   ├── chem/                   # 化学記述子エンジン
│   │   ├── charge_config.py    # 電荷設定データクラス
│   │   ├── protonation.py      # pH依存プロトン化変換
│   │   ├── rdkit_adapter.py    # RDKit + Gasteiger電荷
│   │   ├── xtb_adapter.py      # GFN2-xTB + Mulliken電荷
│   │   ├── mordred_adapter.py  # Mordred 記述子
│   │   ├── unipka_adapter.py   # UniPKa pKa/LogD
│   │   ├── cosmo_adapter.py    # COSMO-RS 溶媒和
│   │   ├── molai_adapter.py    # CNN + PCA 潜在空間
│   │   ├── recommender.py      # 目的変数別推奨記述子DB
│   │   └── smiles_transformer.py
│   ├── data/                   # ロード・前処理・EDA・次元削減
│   ├── models/                 # ML モデル・AutoML・チューナー
│   │   ├── automl.py
│   │   ├── linear_tree.py      # LinearTree/LinearForest/LinearBoost
│   │   ├── rgf.py              # Regularized Greedy Forest
│   │   └── monotonic_kernel.py # 単調制約カーネル
│   ├── pipeline/               # 前処理パイプライン・特徴選択
│   ├── interpret/              # SHAP・SAGE・shapiq
│   └── utils/
├── frontend_nicegui/           # NiceGUI UI（推奨）
├── frontend_streamlit/         # Streamlit UI
├── frontend_django/            # Django REST API
├── tests/                      # テスト
├── tools/                      # セットアップスクリプト・外部バイナリ
│   ├── setup_all.bat           # ★ ワンクリック全自動セットアップ
│   ├── add_xtb_to_path.bat     # xTB PATH設定
│   └── xtb-6.7.1/             # xTBバイナリ（同梱）
├── examples/                   # サンプルスクリプト
├── environment.yml             # Conda環境定義
├── requirements.txt            # pip依存関係
└── pyproject.toml              # ビルド設定
```

---

## 🚀 セットアップ

### 方法A: ワンクリック自動セットアップ（推奨）

> [!TIP]
> **初めての方は `tools\setup_all.bat` をダブルクリックするだけ**で、
> 必須パッケージ → オプショナルパッケージ → xTB → 解釈性ツール → 動作確認
> まで全て自動で行います。一部のインストールが失敗してもアプリは起動できます。

```powershell
# リポジトリをクローン
git clone https://github.com/jckkvs/chemai2.git
cd chemai2

# ワンクリックセットアップ（PowerShellまたはダブルクリック）
.\tools\setup_all.bat
```

---

### 方法B: 手動セットアップ（新規仮想環境を作成する場合）

#### B-1. Conda を使う場合

```powershell
# 1. リポジトリをクローン
git clone https://github.com/jckkvs/chemai2.git
cd chemai2

# 2. Conda 環境を作成 & アクティベート
conda env create -f environment.yml
conda activate ml_gui_app

# 3. オプショナルパッケージをインストール
#    （失敗してもアプリは動作します。失敗したエンジンはUI上でグレーアウトされます）
pip install mordred
pip install unipka
pip install ase
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git
pip install sage-importance shapiq imodels lime eli5
pip install scikit-fingerprints padelpy descriptastorus mol2vec molfeat chemprop
pip install fairchem-core

# 4. xTB PATH設定（量子化学記述子を使う場合）
.\tools\add_xtb_to_path.bat

# 5. アプリ起動
python frontend_nicegui/main.py
```

#### B-2. venv を使う場合

```powershell
# 1. リポジトリをクローン
git clone https://github.com/jckkvs/chemai2.git
cd chemai2

# 2. 仮想環境を作成 & アクティベート
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 必須パッケージをインストール
pip install -r requirements.txt

# 4. オプショナルパッケージをインストール
#    （失敗してもアプリは動作します。失敗したエンジンはUI上でグレーアウトされます）
pip install mordred
pip install unipka
pip install ase
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git
pip install sage-importance shapiq imodels lime eli5
pip install scikit-fingerprints padelpy descriptastorus mol2vec molfeat chemprop
pip install fairchem-core

# 5. xTB PATH設定（量子化学記述子を使う場合）
.\tools\add_xtb_to_path.bat

# 6. アプリ起動
python frontend_nicegui/main.py
```

> [!NOTE]
> venv の場合、RDKit は `pip install rdkit-pypi` でインストール可能ですが、
> Conda (`conda install -c conda-forge rdkit`) の方が安定です。

---

### 方法C: 既存の仮想環境に追加インストールする場合

既にPythonの仮想環境（conda, venv, pyenv等）をお持ちの方向けです。

```powershell
# 1. 既存環境をアクティベート（例: conda）
conda activate my_existing_env

# 2. リポジトリをクローン
git clone https://github.com/jckkvs/chemai2.git
cd chemai2

# 3. 必須パッケージをインストール
#    既存環境のパッケージと競合する場合は個別にバージョン調整してください
pip install -r requirements.txt

# 4. オプショナルパッケージ（失敗してもアプリは動きます）
pip install mordred
pip install unipka
pip install ase
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git
pip install sage-importance shapiq imodels lime eli5
pip install scikit-fingerprints padelpy descriptastorus mol2vec molfeat chemprop
pip install fairchem-core

# 5. xTB PATH設定（量子化学記述子を使う場合）
.\tools\add_xtb_to_path.bat

# 6. アプリ起動
python frontend_nicegui/main.py
```

> [!WARNING]
> 既存環境では numpy / scipy / scikit-learn 等のバージョン競合が起こる場合があります。
> `pip install -r requirements.txt` が失敗した場合は、個別にインストールしてください。

---

## ▶️ アプリ起動

3 つのフロントエンドが利用可能です:

| 版 | コマンド | ポート | 特徴 |
|---|---------|-------|------|
| **NiceGUI** ⭐ | `python frontend_nicegui/main.py` | **8085** | Pure Python UI、2クリック解析 |
| Streamlit | `streamlit run frontend_streamlit/app.py` | 8501 | データ分析向け |
| Django | `python frontend_django/manage.py runserver` | 8000 | REST API、ユーザー認証 |

```powershell
# NiceGUI 版（推奨）
python frontend_nicegui/main.py
# → http://localhost:8085

# Streamlit 版
cd frontend_streamlit
streamlit run app.py
# → http://localhost:8501
```

> [!TIP]
> 各フロントエンドの詳細は以下を参照:
> - [NiceGUI 版 README](frontend_nicegui/README.md) — 起動・設定・デプロイガイド
> - [バックエンド README](backend/README.md) — API ドキュメント

---

## 🧪 提供される記述子エンジン

| エンジン | アダプター | インストール | 主な記述子 | 必須？ |
|----------|-----------|-------------|-----------|--------|
| RDKit | `RDKitAdapter` | 標準搭載 | 分子量・LogP・FP・Gasteiger電荷 | ✅ 必須 |
| GroupContrib | `GroupContribAdapter` | 標準搭載 | 基団寄与法熱物性 | ✅ 必須 |
| Mordred | `MordredAdapter` | `pip install mordred` | 1800+ QSAR記述子 | オプション |
| **xTB** | `XTBAdapter` | tools/xtb-6.7.1 に同梱 | HOMO/LUMO/双極子/Mulliken電荷 | オプション |
| UniPKa | `UniPkaAdapter` | `pip install unipka` | pKa / LogD / 溶媒和エネルギー | オプション |
| COSMO-RS | `CosmoAdapter` | `pip install git+...` | 溶媒和自由エネルギー | オプション |
| MolAI | `MolAIAdapter` | `pip install torch` | CNN+PCA分子潜在空間 | オプション |
| scikit-FP | `SkfpAdapter` | `pip install scikit-fingerprints` | 30+種フィンガープリント | オプション |
| Mol2Vec | `Mol2VecAdapter` | `pip install mol2vec` | Word2Vec分子埋め込み | オプション |
| Molfeat | `MolfeatAdapter` | `pip install molfeat` | 統合FPフレームワーク | オプション |
| Chemprop | `ChempropAdapter` | `pip install chemprop` | D-MPNN GNN | オプション |
| PaDEL | `PaDELAdapter` | `pip install padelpy` | 1800+記述子(Java必要) | オプション |

> [!IMPORTANT]
> オプションのエンジンがインストールされていなくてもアプリは正常に起動します。
> 未インストールのエンジンはUI上でグレーアウト表示され、利用可能なエンジンのみで解析できます。

---

## 🧪 テスト

```powershell
# 全テスト実行
python -m pytest tests/ -v --tb=short

# 電荷設定モジュールのみ
python -m pytest tests/test_charge_config.py tests/test_charge_extended.py -v

# カバレッジ付き
python -m pytest tests/ --cov=backend --cov-branch --cov-report=term-missing
```

---

## 📝 サンプル

```powershell
python examples/descriptor_calculation_example.py
```

> [!NOTE]
> 詳しい再現手順は [REPRODUCE.md](REPRODUCE.md) を参照してください。
