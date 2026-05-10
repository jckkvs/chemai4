# Backend Module

`backend/` は ChemAI ML Studio のコアエンジンを含みます。すべてのフロントエンド（Streamlit / NiceGUI / Django）から共通で利用されます。

---

## モジュール構成

### 1. Data (`backend/data/`)

データの読み込み・前処理・探索的データ分析を担当します。

| ファイル | 説明 |
|---------|------|
| `loader.py` | CSV / Excel / Parquet / JSON / SQLite / SDF / MOL からのデータ読み込み・保存 |
| `type_detector.py` | 各列の変数型（数値/カテゴリ/バイナリ/定数/SMILES/日時/テキスト/周期）を自動判定 |
| `preprocessor.py` | `ColumnTransformer` を利用した変数型別前処理パイプライン自動構築 |
| `feature_engineer.py` | 交互作用特徴量 / グループ集約 / 日時特徴量抽出 / ラグ・ローリング特徴量 |
| `data_cleaner.py` | 列削除 / 欠損行除去 / 定数列除去 / 外れ値クリッピング / 重複行除去 |
| `eda.py` | 基本統計量 / 相関分析 / 外れ値検出 / 分布可視化 / 目的変数分析 |
| `dim_reduction.py` | 次元削減（PCA / t-SNE / UMAP） |
| `leakage_detector.py` | 学習データ・テストデータ間のリーケージリスク検出 |
| `benchmark.py` | モデルベンチマーク・評価指標計算 |
| `benchmark_datasets.py` | ESOL / FreeSolv / Lipophilicity のダウンロード・キャッシュ |

### 2. Models (`backend/models/`)

機械学習モデルの構築・最適化・評価を担当します。

| ファイル | 説明 |
|---------|------|
| `factory.py` | モデルレジストリ（Ridge / RF / XGBoost / LightGBM / CatBoost / SVR 等のファクトリ） |
| `automl.py` | ワンボタンAutoML実行エンジン（タスク自動判定→前処理→全モデル比較→最適選択） |
| `tuner.py` | Optuna によるハイパーパラメータ最適化 |
| `cv_manager.py` | CV戦略管理（KFold / Stratified / Group / WalkForward / Repeated） |
| `cv_bias_evaluator.py` | Tibshirani / BBC-CV によるCV偏りバイアス補正 |
| `linear_tree.py` | LinearTree / LinearForest / LinearBoost（フルスクラッチ実装） |
| `rgf.py` | Regularized Greedy Forest（フルスクラッチ実装） |
| `monotonic_kernel.py` | カーネルモデルへのソフト単調性制約ラッパー |

### 3. Pipeline (`backend/pipeline/`)

scikit-learn Pipeline の組立・グリッド探索を担当します。

| ファイル | 説明 |
|---------|------|
| `pipeline_builder.py` | 入力列選択→前処理→特徴量生成→特徴量選択→推定器 の5段Pipeline構築 |
| `pipeline_grid.py` | 各ステップの複数候補からデカルト積でPipeline候補を生成 |
| `column_selector.py` | mlxtend ラッパー、列メタ情報（単調性/グループ）管理 |
| `col_preprocessor.py` | 変数型ルール別の前処理Transformer |
| `feature_generator.py` | 多項式・交互作用特徴量生成Transformer |
| `feature_selector.py` | Lasso / RF / SelectKBest / Boruta / ReliefF 等の特徴量選択 |

### 4. Interpret (`backend/interpret/`)

モデル解釈・説明性を担当します。

| ファイル | 説明 |
|---------|------|
| `shap_explainer.py` | SHAP（Tree / Linear / Kernel / Deep）+ 各種プロット |
| `sri.py` | SHAP SRI分解（Synergy / Redundancy / Independence） |

### 5. Chem (`backend/chem/`)

化学情報学に特化した記述子計算を担当します。

| ファイル | 説明 |
|---------|------|
| `base.py` | `BaseChemAdapter` 抽象基底クラス |
| `rdkit_adapter.py` | RDKit 記述子（200種類+ / フィンガープリント） |
| `xtb_adapter.py` | GFN2-xTB 量子化学記述子（HOMO/LUMO/双極子モーメント等） |
| `mordred_adapter.py` | Mordred 1,800+記述子 |
| `uma_adapter.py` | Meta UMA (fairchem) 学習済み分子表現 |
| `cosmo_adapter.py` | COSMO-RS 溶媒和エネルギー |
| `unipka_adapter.py` | UniPKa pKa/LogD予測 |
| `group_contrib_adapter.py` | 基団寄与法 |
| `molai_adapter.py` | CNN+PCA 潜在ベクトル |
| `smiles_transformer.py` | SMILES→記述子DataFrame変換パイプライン |
| `charge_config.py` | 分子電荷・スピン多重度設定 |
| `protonation.py` | pH依存プロトン化状態の適用 |

### 6. Optim (`backend/optim/`)

ベイズ最適化・探索空間定義を担当します。

| ファイル | 説明 |
|---------|------|
| `bayesian_optimizer.py` | ガウス過程ベースのベイズ最適化エンジン |
| `search_space.py` | 連続/離散/カテゴリ変数の探索空間定義 |
| `constraints.py` | 制約条件管理（線形/非線形/範囲制約） |
| `bo_visualizer.py` | 最適化履歴・獲得関数の可視化 |

### 7. Utils (`backend/utils/`)

共通ユーティリティを提供します。

| ファイル | 説明 |
|---------|------|
| `config.py` | グローバル設定（RANDOM_STATE / パス / AutoML / SHAP / MLflow） |
| `optional_import.py` | 安全import（未インストールライブラリのフォールバック） |
| `param_schema.py` | ハイパーパラメータスキーマ定義 |

---

## 主要アルゴリズム

### 次元削減
- **PCA**: 線形な分散最大化方向への投影
- **t-SNE**: 高次元の局所的な構造を維持する非線形埋め込み
- **UMAP**: 局所構造と大局構造のバランスに優れた高速非線形埋め込み

### フルスクラッチ実装モデル
- **LinearTree / LinearForest / LinearBoost**: 葉にリニアモデルを持つ決定木系アンサンブル
- **RGF (Regularized Greedy Forest)**: 正則化付き貪欲森林（L1/L2正則化対応）

### CV偏りバイアス補正
- **Tibshirani法**: 楽観バイアスの推定
- **BBC-CV (Bootstrap Bias Corrected CV)**: ブートストラップによる偏り補正

---

## 開発者向け情報

### 遅延インポート
`backend.utils.optional_import` を使用して、重いライブラリ（UMAP, Mordred, RDKit 等）は必要になるまでインポートされません。

### テスト
```bash
python -m pytest tests/ -v --tb=short
```

### 拡張方法
詳細は [CONTRIBUTING_GUIDE.md](../CONTRIBUTING_GUIDE.md) を参照してください。
