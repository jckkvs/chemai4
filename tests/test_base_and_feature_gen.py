"""
tests/test_base_and_feature_gen.py

BaseChemAdapter / DescriptorResult / DescriptorMetadata /
FeatureGenerator / FeatureGenConfig の包括テスト。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.chem.base import (
    BaseChemAdapter,
    DescriptorResult,
    DescriptorMetadata,
)
from backend.pipeline.feature_generator import (
    FeatureGenConfig,
    FeatureGenerator,
)


# ============================================================
# DescriptorResult テスト
# ============================================================

class TestDescriptorResult:
    def test_success_rate_all_success(self):
        result = DescriptorResult(
            descriptors=pd.DataFrame({"a": [1, 2, 3]}),
            smiles_list=["CCO", "c1ccccc1", "CC"],
            failed_indices=[],
            adapter_name="test",
        )
        assert result.success_rate == pytest.approx(1.0)

    def test_success_rate_partial(self):
        result = DescriptorResult(
            descriptors=pd.DataFrame({"a": [1, 2]}),
            smiles_list=["CCO", "c1ccccc1", "XX"],
            failed_indices=[2],
            adapter_name="test",
        )
        assert result.success_rate == pytest.approx(2 / 3)

    def test_success_rate_empty(self):
        result = DescriptorResult(
            descriptors=pd.DataFrame(),
            smiles_list=[],
            failed_indices=[],
            adapter_name="test",
        )
        assert result.success_rate == 0.0

    def test_n_descriptors(self):
        result = DescriptorResult(
            descriptors=pd.DataFrame({"a": [1], "b": [2], "c": [3]}),
            smiles_list=["CCO"],
            failed_indices=[],
            adapter_name="test",
        )
        assert result.n_descriptors == 3


# ============================================================
# DescriptorMetadata テスト
# ============================================================

class TestDescriptorMetadata:
    def test_basic(self):
        meta = DescriptorMetadata(
            name="MolWt",
            meaning="分子量",
            is_count=False,
            description="分子の総質量（Da）",
        )
        assert meta.name == "MolWt"
        assert meta.is_binary is False

    def test_binary(self):
        meta = DescriptorMetadata(
            name="HasRing",
            meaning="環構造の有無",
            is_count=False,
            is_binary=True,
        )
        assert meta.is_binary is True


# ============================================================
# BaseChemAdapter テスト（抽象クラスのサブクラス）
# ============================================================

class DummyAdapter(BaseChemAdapter):
    """テスト用のダミーアダプタ。"""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "テスト用ダミーアダプタ"

    def is_available(self) -> bool:
        return True

    def compute(self, smiles_list, **kwargs):
        df = pd.DataFrame({"mw": [100.0] * len(smiles_list)})
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=[],
            adapter_name=self.name,
        )


class UnavailableAdapter(BaseChemAdapter):
    @property
    def name(self) -> str:
        return "unavailable"

    @property
    def description(self) -> str:
        return "利用不可アダプタ"

    def is_available(self) -> bool:
        return False

    def compute(self, smiles_list, **kwargs):
        self._require_available()


class TestBaseChemAdapter:
    def test_repr_available(self):
        a = DummyAdapter()
        assert "✓ available" in repr(a)

    def test_repr_unavailable(self):
        a = UnavailableAdapter()
        assert "✗ unavailable" in repr(a)

    def test_require_available_raises(self):
        a = UnavailableAdapter()
        with pytest.raises(RuntimeError, match="インストールされていません"):
            a.compute(["CCO"])

    def test_get_descriptor_names_default(self):
        a = DummyAdapter()
        assert a.get_descriptor_names() == []

    def test_get_descriptors_metadata_default(self):
        a = DummyAdapter()
        assert a.get_descriptors_metadata() == []

    def test_compute(self):
        a = DummyAdapter()
        result = a.compute(["CCO", "c1ccccc1"])
        assert result.n_descriptors == 1
        assert result.success_rate == 1.0


# ============================================================
# FeatureGenConfig テスト
# ============================================================

class TestFeatureGenConfig:
    def test_defaults(self):
        cfg = FeatureGenConfig()
        assert cfg.method == "none"
        assert cfg.degree == 2
        assert cfg.include_bias is False


# ============================================================
# FeatureGenerator テスト
# ============================================================

class TestFeatureGenerator:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    def test_passthrough(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        gen.fit(sample_df)
        result = gen.transform(sample_df)
        assert result.shape == (3, 2)
        assert gen.is_passthrough

    def test_polynomial(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        gen.fit(sample_df)
        result = gen.transform(sample_df)
        # degree=2, no bias: a, b, a^2, ab, b^2 = 5 features
        assert result.shape[1] == 5

    def test_interaction_only(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="interaction_only", degree=2))
        gen.fit(sample_df)
        result = gen.transform(sample_df)
        # interaction_only: a, b, ab = 3 features
        assert result.shape[1] == 3

    def test_get_feature_names_out_passthrough(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        gen.fit(sample_df)
        names = gen.get_feature_names_out()
        assert list(names) == ["a", "b"]

    def test_get_feature_names_out_polynomial(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        gen.fit(sample_df)
        names = gen.get_feature_names_out()
        assert len(names) == 5

    def test_n_output_features_passthrough(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        gen.fit(sample_df)
        assert gen.n_output_features == 2

    def test_n_output_features_polynomial(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        gen.fit(sample_df)
        assert gen.n_output_features == 5

    def test_numpy_input(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        gen.fit(X)
        result = gen.transform(X)
        assert result.shape[1] == 5

    def test_with_bias(self, sample_df):
        gen = FeatureGenerator(FeatureGenConfig(
            method="polynomial", degree=2, include_bias=True
        ))
        gen.fit(sample_df)
        result = gen.transform(sample_df)
        # bias + a, b, a^2, ab, b^2 = 6 features
        assert result.shape[1] == 6
