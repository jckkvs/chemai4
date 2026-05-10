"""
tests/test_e2e_smiles_to_ml.py

DOMサンプルデータ読込 → SMILES特徴量変換 → 機械学習 の完全E2Eテスト。

対応フロー:
  1. data_tab.py の _load_sample_regression / _load_sample_classification と
     同じサンプルデータを生成
  2. SmilesDescriptorTransformer でRDKit記述子を計算
  3. AutoMLEngine.run() でCross-Validation + 最良モデル選択
  4. 結果の検証（スコア・予測値・例外なし）

DoD:
  - 全テストがPASS
  - best_model_key が返る
  - predict() がNaNなしの予測値を返す
  - エラー・例外なしで完走
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── サンプルSMILES（data_tab.py SAMPLE_SMILES と同一） ──
SAMPLE_SMILES = [
    "C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "c1ccccc1O",
    "CC(=O)O", "CC(C)C", "C1CCCCC1", "c1ccncc1", "c1ncncn1", "C1COCCO1",
    "CC(=O)OC", "CCOC", "CCOCC", "CC(O)CC", "c1ccc(Cl)cc1",
    "CC(=O)N", "CCCCCO", "c1ccc(F)cc1", "CC(C)=O", "OCCO",
    "CC(=O)CC", "CCCCO",
]


# ============================================================
# データ生成ヘルパー（UI の _load_sample_* と同等）
# ============================================================

def _make_regression_df(n: int = 25, seed: int = 42) -> pd.DataFrame:
    """回帰用サンプルデータ（SMILES + solubility_logS）"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SMILES": rng.choice(SAMPLE_SMILES, n),
        "solubility_logS": rng.standard_normal(n) * 2 - 2,
    })


def _make_classification_df(n: int = 25, seed: int = 42) -> pd.DataFrame:
    """分類用サンプルデータ（SMILES + is_toxic: 0/1）"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SMILES": rng.choice(SAMPLE_SMILES, n),
        "is_toxic": rng.integers(0, 2, n),
    })


# ============================================================
# F-01: SMILES記述子変換テスト
# ============================================================

class TestSmilesDescriptorTransform:
    """F-01: SmilesDescriptorTransformer の変換動作を検証する。"""

    def test_rdkit_descriptors_computed(self):
        """
        T-01: RDKit記述子が1つ以上計算されること。
        Implements: F-01 | SMILES→記述子変換の基本動作

        注意: SmilesDescriptorTransformer.transform()はSMILS列を除去して記述子列に置換する。
        これは意図された動作であり、SMILES列はお子文字列のままskleanモデルに渡せないため。
        """
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _make_regression_df(15)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        # SMILES列は記述子に変換され除去される（正しい動作）
        assert "SMILES" not in df_out.columns, (
            "SMILES列は記述子変換後に除去されるはず"
        )
        # 目的変数列は保持される（DataFrame全体に渡せば保持）
        assert "solubility_logS" in df_out.columns, "目的変数列が失われている"

        desc_cols = [c for c in df_out.columns if c != "solubility_logS"]
        assert len(desc_cols) >= 1, f"記述子が1つも計算されていない: {df_out.columns.tolist()}"

    def test_no_all_nan_columns(self):
        """
        T-02: 計算した記述子に全行NaNの列がないこと。
        (fit()内で全NaN列は自動除去されるので、実際にないはず)
        """
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _make_regression_df(15)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        # SMILES列と目的変数列を除いた記述子列に全NaNがないこと
        desc_cols = [c for c in df_out.columns if c != "solubility_logS"]
        for col in desc_cols:
            na_rate = df_out[col].isna().mean()
            assert na_rate < 1.0, f"列 '{col}' が全行NaN"

    def test_row_count_preserved(self):
        """
        T-03: 変換後の行数が元データと一致すること。
        """
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer

        df = _make_regression_df(25)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_out = transformer.fit_transform(df)

        assert len(df_out) == len(df), (
            f"行数が変わっている: {len(df)} -> {len(df_out)}"
        )


# ============================================================
# F-02: SMILES→記述子→AutoML 完全E2Eパイプライン（回帰）
# ============================================================

class TestE2ERegressionSmilesToML:
    """F-02: サンプル回帰データのE2E完走を検証する。"""

    @pytest.fixture(scope="class")
    def regression_result(self):
        """E2E実行結果（クラス内で共有）"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.models.automl import AutoMLEngine

        # Step 1: サンプルデータ生成（UIの _load_sample_regression と同等）
        df = _make_regression_df(25)
        assert "SMILES" in df.columns
        assert "solubility_logS" in df.columns

        # Step 2: SMILES記述子変換
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        # 記述子が計算されたことを確認
        extra_cols = [c for c in df_desc.columns if c not in ("SMILES", "solubility_logS")]
        assert len(extra_cols) >= 1, "記述子が計算されなかった"

        # Step 3: AutoML実行（軽量モデル2つ、CV=2）
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge", "rf"],
            timeout_seconds=120,
        )
        result = engine.run(df_desc, target_col="solubility_logS")
        return result, df_desc

    def test_e2e_completes_without_error(self, regression_result):
        """
        T-04: エラーなしで完走すること。
        Implements: F-02 | E2E完走の最基本条件
        """
        result, _ = regression_result
        assert result is not None, "AutoMLResultがNone"

    def test_best_model_selected(self, regression_result):
        """
        T-05: 最良モデルが選択されていること。
        """
        from backend.models.automl import AutoMLResult

        result, _ = regression_result
        assert isinstance(result, AutoMLResult), f"戻り値がAutoMLResultではない: {type(result)}"
        assert result.best_model_key in ("ridge", "rf"), (
            f"best_model_key が期待外: {result.best_model_key}"
        )

    def test_best_score_is_finite(self, regression_result):
        """
        T-06: best_scoreが有限値であること。
        """
        result, _ = regression_result
        assert np.isfinite(result.best_score), (
            f"best_score が有限値でない: {result.best_score}"
        )

    def test_best_pipeline_predict(self, regression_result):
        """
        T-07: 最良パイプラインで予測が実行でき、NaNがないこと。
        """
        result, df_desc = regression_result
        X = df_desc.drop(columns=["solubility_logS"])
        preds = result.best_pipeline.predict(X)

        assert len(preds) == len(df_desc), (
            f"予測数が不一致: {len(preds)} vs {len(df_desc)}"
        )
        assert not np.isnan(preds).any(), "予測値にNaNが含まれる"

    def test_model_scores_populated(self, regression_result):
        """
        T-08: model_scoresに全モデルのスコアが格納されていること。
        """
        result, _ = regression_result
        assert len(result.model_scores) >= 1, "model_scoresが空"
        for key, score in result.model_scores.items():
            assert np.isfinite(score), f"モデル '{key}' のスコアが非有限値: {score}"

    def test_elapsed_is_positive(self, regression_result):
        """
        T-09: 経過時間が正の値であること。
        """
        result, _ = regression_result
        assert result.elapsed_seconds > 0, f"elapsed_secondsが非正: {result.elapsed_seconds}"


# ============================================================
# F-03: SMILES→記述子→AutoML 完全E2Eパイプライン（分類）
# ============================================================

class TestE2EClassificationSmilesToML:
    """F-03: サンプル分類データのE2E完走を検証する。"""

    @pytest.fixture(scope="class")
    def classification_result(self):
        """分類E2E実行結果（クラス内で共有）"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.models.automl import AutoMLEngine

        # Step 1: 分類サンプルデータ生成
        df = _make_classification_df(25)

        # Step 2: SMILES記述子変換
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        # Step 3: AutoML実行（分類）
        engine = AutoMLEngine(
            task="classification",
            cv_folds=2,
            model_keys=["dt_c"],
            timeout_seconds=120,
        )
        result = engine.run(df_desc, target_col="is_toxic")
        return result, df_desc

    def test_e2e_classification_completes(self, classification_result):
        """
        T-10: 分類E2Eがエラーなしで完走すること。
        """
        result, _ = classification_result
        assert result is not None

    def test_classification_task_detected(self, classification_result):
        """
        T-11: タスク種別が「classification」と判定されていること。
        """
        result, _ = classification_result
        assert result.task == "classification", (
            f"task が classification でない: {result.task}"
        )

    def test_classification_predict(self, classification_result):
        """
        T-12: 分類予測がNaNなしで返ること。
        """
        result, df_desc = classification_result
        X = df_desc.drop(columns=["is_toxic"])
        preds = result.best_pipeline.predict(X)
        assert len(preds) == len(df_desc)
        assert not np.isnan(preds.astype(float)).any(), "分類予測にNaNが含まれる"


# ============================================================
# F-04: 自動タスク判定（auto モード）
# ============================================================

class TestAutoTaskDetection:
    """F-04: task='auto' で回帰/分類が自動判定されること。"""

    def test_auto_detects_regression(self):
        """
        T-13: float目的変数のとき 'regression' が自動選択されること。
        """
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.models.automl import AutoMLEngine

        df = _make_regression_df(20)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        engine = AutoMLEngine(task="auto", cv_folds=2, model_keys=["ridge"], timeout_seconds=60)
        result = engine.run(df_desc, target_col="solubility_logS")
        assert result.task == "regression"

    def test_auto_detects_classification(self):
        """
        T-14: int(0/1)目的変数のとき 'classification' が自動選択されること。
        """
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.models.automl import AutoMLEngine

        df = _make_classification_df(25)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        engine = AutoMLEngine(task="auto", cv_folds=2, model_keys=["dt_c"], timeout_seconds=60)
        result = engine.run(df_desc, target_col="is_toxic")
        assert result.task == "classification"


# ============================================================
# F-05: smiles_col指定でのAutoMLエンジン直接利用
# ============================================================

class TestAutoMLWithSmilesColDirect:
    """F-05: smiles_col をAutoMLEngineに直接渡した場合の動作検証。"""

    def test_smiles_col_in_engine_run(self):
        """
        T-15: AutoMLEngine に smiles_col='SMILES' を渡したとき正常動作すること。
        Implements: F-05 | analysis_runner.py のフロー模倣
        """
        from backend.models.automl import AutoMLEngine

        df = _make_regression_df(20)
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"],
            timeout_seconds=60,
        )
        result = engine.run(df, target_col="solubility_logS", smiles_col="SMILES")
        assert result is not None
        assert result.best_model_key is not None
        preds = result.best_pipeline.predict(df.drop(columns=["solubility_logS"]))
        assert len(preds) == len(df)


# ============================================================
# F-06: 記述子計算後の欠損値処理（前処理パイプライン）
# ============================================================

class TestDescriptorNaNHandling:
    """F-06: 記述子にNaNが含まれていても前処理で補完されること。"""

    def test_nan_imputed_in_pipeline(self):
        """
        T-16: 記述子にNaNが含まれていてもAutoMLが完走すること（imputation動作確認）。
        """
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.models.automl import AutoMLEngine

        df = _make_regression_df(20)
        transformer = SmilesDescriptorTransformer(smiles_col="SMILES")
        df_desc = transformer.fit_transform(df)

        # 記述子列に意図的にNaNを注入
        desc_cols = [c for c in df_desc.columns if c not in ("SMILES", "solubility_logS")]
        if desc_cols:
            df_desc.loc[df_desc.index[:3], desc_cols[0]] = np.nan

        engine = AutoMLEngine(
            task="regression", cv_folds=2, model_keys=["ridge"], timeout_seconds=60,
        )
        # NaN補完がパイプライン内で行われ、エラーなく完走すること
        result = engine.run(df_desc, target_col="solubility_logS")
        assert result is not None
        assert result.best_model_key is not None
