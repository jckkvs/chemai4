# EVIDENCE: ChemAI Analytical Suite The Missing 38 Features
## 1. 原典の引用と翻訳 (Source References)

本プロジェクトにおける「The Missing 38 Features」は、システム基盤の完全性保証および論文等で提示された高度な分析・実装要求を果たすための仕様となります。
各機能の実装にあたり、要求仕様（PROMPT）と実装の対応状況を以下に示します。

### 1.1 単調性制約 (Monotonic Constraints)
> **Source:** "Build a complete monotonic constraint wrapper that supports advanced models including XGBoost, LightGBM, and generic Sklearn estimators."
> **翻訳:** "XGBoost, LightGBM, および一般的なSklearn推定器をサポートする完全な単調性制約ラッパーを構築せよ。"
- **コード対応:** `backend/models/monotonic_wrapper.py`: `wrap_monotonic()`
- **テストID:** T-228 (`tests/test_monotonic_wrapper.py`)

### 1.2 教師付き特徴量選択 (Supervised Feature Selection)
> **Source:** "Feature Selector must support various approaches: Lasso, Ridge, Random Forests, ReliefF, Boruta, Genetic Selection, and GroupLasso, ensuring robustness across classification and regression."
> **翻訳:** "特徴量セレクタはLasso, Ridge, Random Forests, ReliefF, Boruta, Genetic Selection, GroupLassoなどの多様な手法をサポートし、回帰および分類タスク間での堅牢性を保証しなければならない。"
- **コード対応:** `backend/pipeline/feature_selector.py`: `FeatureSelector._build_selector()`
- **テストID:** T-701 ~ T-715 (`tests/test_feature_selector.py`, `tests/test_feature_selector_success.py`)

## 2. コードとテスト対応表 (Code to Test Traceability)

FeatureMatrix.md において各機能（F-xxx）とテスト（T-xxx）の1:1対応を定義しています。
主なモジュールの対応表を抜粋します。

| 要求仕様 (Feature) | 実装箇所 (Code Line / Function) | テストID (Test ID) |
| :--- | :--- | :--- |
| F-211 ~ 220 (Linear Tree) | `backend/models/linear_tree.py` 全般 | T-211 ~ T-220 |
| F-225 ~ 228 (Monotonic) | `backend/models/monotonic_wrapper.py` | T-225 ~ T-228 |
| F-229 ~ 231 (RGF) | `backend/models/rgf.py` | T-229 ~ T-231 |
| F-101 ~ 110 (AutoML) | `backend/models/automl.py` | T-101 ~ T-110 |

## 3. 再現性確保 (Reproducibility)

- **環境定義:** 実行環境は `pyproject.toml` にて厳密に定義されています。オプショナルな依存関係 (`[project.optional-dependencies]`) はフォールバックロジックで対応、またはフルインストール時に動作を保証します。
- **固定シード:** 全ての確率的処理（RandomForest, XGBoost, etc.）は `backend.utils.config.RANDOM_STATE` (デフォルト = 42) を使用して固定されています。
- **データ前処理:** `backend/data/preprocessor.py` および `backend/pipeline/col_preprocessor.py` にて欠損値補完、One-Hotエンコーディング等の前処理の決定性を確保しています。

## 4. ベンチマーク・カバレッジ結果

カバレッジは `full_coverage_utf8.txt` に記録された通り、全実装対象コード（非対象オプショナルドメインを除く）について行カバレッジ ≥ 90% を達成しています。
分岐カバレッジについてもpytest-covの `--cov-branch` を用いて測定され、基準を十分満たしています。
