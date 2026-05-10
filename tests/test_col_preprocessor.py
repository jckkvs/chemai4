# -*- coding: utf-8 -*-
"""
tests/test_col_preprocessor.py

ColPreprocessor の完全なテストスイート。
ColPreprocessConfig の各パラメータ組み合わせにおけるパルプラインの構築や、
TypeDetector のオーバーライド指定、様々なスケーラー・エンコーダー・インピューターの動作を検証する。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from backend.pipeline.col_preprocessor import ColPreprocessConfig, ColPreprocessor

# ────────────────────────────────────────────────────────────
# フィクスチャ
# ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """
    数値、バイナリ、低カーディナリティ、高カーディナリティの各列を含むサンプルデータ。
    """
    df = pd.DataFrame({
        "num1": [1.0, 2.5, 3.1, np.nan, 5.0, 10.0, 1.2, 5.0, 3.3, 1.1],
        "num2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "bin1": [0, 1, 0, 1, 0, 1, np.nan, 0, 1, 0],
        "cat_low": ["A", "A", "B", "B", "C", "A", "C", "C", np.nan, "A"],
        "cat_high": ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10"]
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return df, y

# ────────────────────────────────────────────────────────────
# テスト
# ────────────────────────────────────────────────────────────

class TestColPreprocessor:
    def test_default_config_fit_transform(self, sample_data):
        X, y = sample_data
        cp = ColPreprocessor()
        cp.fit(X, y)
        X_trans = cp.transform(X)
        
        # 欠損値がないこと
        assert not np.any(pd.isna(X_trans))
        # 出力の特徴量名が取得できること
        assert len(cp.get_feature_names_out()) > 0
        assert X_trans.shape[0] == 10

    def test_numeric_imputers(self, sample_data):
        X, y = sample_data
        imputers = ["mean", "median", "knn", "iterative", "constant", "most_frequent"]
        
        for imp in imputers:
            cfg = ColPreprocessConfig(numeric_imputer=imp, constant_fill_value=-999)
            cp = ColPreprocessor(cfg)
            cp.fit(X, y)
            X_trans = cp.transform(X)
            assert not np.any(pd.isna(X_trans))

    def test_numeric_scalers(self, sample_data):
        X, y = sample_data
        
        # negative values not allowed for power_bc
        X_pos = X.copy()
        X_pos["num1"] = X_pos["num1"].fillna(5.0) + 10.0 
        
        scalers = [
            "standard", "minmax", "robust", "maxabs", 
            "power_yj", "power_bc", "quantile_normal", "quantile_uniform", 
            "log", "none"
        ]
        
        for sc in scalers:
            cfg = ColPreprocessConfig(numeric_scaler=sc)
            cp = ColPreprocessor(cfg)
            cp.fit(X_pos, y)
            X_trans = cp.transform(X_pos)
            assert X_trans.shape[0] == 10

    def test_categorical_low_encoders(self, sample_data):
        encoders = ["onehot", "ordinal", "target", "binary", "woe", "unknown_encoder_low"]

        X, y = sample_data
        for enc in encoders:
            cfg = ColPreprocessConfig(cat_low_encoder=enc)
            cp = ColPreprocessor(cfg)
            cp.fit(X, y)
            X_trans = cp.transform(X)
            assert X_trans.shape[0] == 10

    def test_categorical_high_encoders(self, sample_data):
        encoders = ["target", "hashing", "binary", "leaveoneout", "ordinal", "unknown_encoder_high"]

        X, y = sample_data
        for enc in encoders:
            cfg = ColPreprocessConfig(cat_high_encoder=enc)
            cp = ColPreprocessor(cfg)
            cp.fit(X, y)
            X_trans = cp.transform(X)
            assert X_trans.shape[0] == 10

    def test_binary_encoders_and_imputers(self, sample_data):
        X, y = sample_data
        cfg = ColPreprocessConfig(
            binary_encoder="passthrough", 
            binary_imputer="knn",
        )
        cp = ColPreprocessor(cfg)
        cp.fit(X, y)
        X_trans = cp.transform(X)
        assert X_trans.shape[0] == 10

    def test_override_types(self, sample_data):
        X, y = sample_data
        # 強制的に num1 を categorical_low、cat_high を passthrough に変更
        cfg = ColPreprocessConfig(
            override_types={"num1": "category_low", "cat_high": "passthrough"}
        )
        cp = ColPreprocessor(cfg)
        cp.fit(X, y)
        
        assert "num1" in cp._ct.named_transformers_["cat_low"].feature_names_in_
        # Check how passthrough is handled.
        passthrough_cols = [cols for name, trans, cols in cp._ct.transformers_ if name == "passthrough"][0]
        assert "cat_high" in passthrough_cols

    def test_not_fitted_error(self, sample_data):
        from sklearn.exceptions import NotFittedError
        X, y = sample_data
        cp = ColPreprocessor()
        with pytest.raises((NotFittedError, ValueError, RuntimeError)):
            cp.transform(X)

    def test_no_valid_columns(self):
        # np.nan だけのDFなどはすべてdropされる可能性があるのでダミー列を使う
        X = pd.DataFrame()
        cp = ColPreprocessor()
        with pytest.raises(ValueError, match="前処理対象列が見つかりません"):
            cp.fit(X)

    def test_set_output_pandas(self, sample_data):
        X, y = sample_data
        cp = ColPreprocessor()
        cp.set_output(transform="pandas")
        cp.fit(X, y)
        X_trans = cp.transform(X)
        assert isinstance(X_trans, pd.DataFrame)
        assert "num1" in str(X_trans.columns)
