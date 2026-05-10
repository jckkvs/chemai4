# -*- coding: utf-8 -*-
"""
tests/test_utils_and_base.py

backend/utils と backend/chem/base の包括テスト。
  - config.py: 定数/AppConfig
  - optional_import.py: safe_import/is_available/require/probe_all
  - base.py: DescriptorResult/DescriptorMetadata/BaseChemAdapter
  - benchmark_datasets.py: list_benchmark_datasets
  - molai_adapter.py: MolAIAdapter (torch依存)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════
# config.py テスト
# ═══════════════════════════════════════════════════════════════════

class TestConfig:

    def test_random_state_exists(self):
        from backend.utils.config import RANDOM_STATE
        assert isinstance(RANDOM_STATE, int)
        assert RANDOM_STATE == 42

    def test_project_root_is_dir(self):
        from backend.utils.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_app_config_defaults(self):
        from backend.utils.config import AppConfig
        cfg = AppConfig()
        assert cfg.random_state == 42
        assert cfg.automl_cv_folds == 5
        assert cfg.n_jobs == -1
        assert isinstance(cfg.extra, dict)

    def test_app_config_custom(self):
        from backend.utils.config import AppConfig
        cfg = AppConfig(random_state=123, automl_cv_folds=10)
        assert cfg.random_state == 123
        assert cfg.automl_cv_folds == 10

    def test_default_config_instance(self):
        from backend.utils.config import default_config
        assert default_config is not None
        assert default_config.random_state == 42

    def test_threshold_constants(self):
        from backend.utils.config import (
            TYPE_DETECTOR_CARDINALITY_THRESHOLD,
            TYPE_DETECTOR_SKEWNESS_THRESHOLD,
            TYPE_DETECTOR_OUTLIER_IQR_FACTOR,
        )
        assert TYPE_DETECTOR_CARDINALITY_THRESHOLD > 0
        assert TYPE_DETECTOR_SKEWNESS_THRESHOLD > 0
        assert TYPE_DETECTOR_OUTLIER_IQR_FACTOR > 0

    def test_automl_constants(self):
        from backend.utils.config import AUTOML_CV_FOLDS, AUTOML_TIMEOUT_SECONDS
        assert AUTOML_CV_FOLDS >= 2
        assert AUTOML_TIMEOUT_SECONDS > 0


# ═══════════════════════════════════════════════════════════════════
# optional_import.py テスト
# ═══════════════════════════════════════════════════════════════════

class TestOptionalImport:

    def test_safe_import_existing_module(self):
        from backend.utils.optional_import import safe_import
        mod = safe_import("json")
        assert mod is not None

    def test_safe_import_missing_module(self):
        from backend.utils.optional_import import safe_import
        mod = safe_import("nonexistent_module_xyz_123")
        assert mod is None

    def test_safe_import_with_alias(self):
        from backend.utils.optional_import import safe_import, is_available
        safe_import("json", "json_alias_test")
        assert is_available("json_alias_test") is True

    def test_is_available_missing(self):
        from backend.utils.optional_import import is_available
        assert is_available("totally_nonexistent_lib") is False

    def test_require_available(self):
        from backend.utils.optional_import import safe_import, require
        safe_import("json", "json_require_test")
        require("json_require_test")  # 例外なし

    def test_require_unavailable(self):
        from backend.utils.optional_import import require
        with pytest.raises(RuntimeError, match="インストールされていません"):
            require("nonexistent_lib_for_require_test", feature="テスト機能")

    def test_get_availability_report(self):
        from backend.utils.optional_import import get_availability_report, safe_import
        safe_import("os", "os_test")
        report = get_availability_report()
        assert isinstance(report, dict)
        assert "os_test" in report

    def test_probe_all_optional_libraries(self):
        from backend.utils.optional_import import probe_all_optional_libraries
        report = probe_all_optional_libraries()
        assert isinstance(report, dict)
        assert len(report) > 0


# ═══════════════════════════════════════════════════════════════════
# chem/base.py テスト
# ═══════════════════════════════════════════════════════════════════

class TestDescriptorResult:

    def test_success_rate_all_ok(self):
        from backend.chem.base import DescriptorResult
        dr = DescriptorResult(
            descriptors=pd.DataFrame({"a": [1, 2, 3]}),
            smiles_list=["C", "CC", "CCC"],
            failed_indices=[],
            adapter_name="test",
        )
        assert dr.success_rate == 1.0

    def test_success_rate_some_failed(self):
        from backend.chem.base import DescriptorResult
        dr = DescriptorResult(
            descriptors=pd.DataFrame({"a": [1, np.nan, 3]}),
            smiles_list=["C", "CC", "CCC"],
            failed_indices=[1],
            adapter_name="test",
        )
        assert abs(dr.success_rate - 2 / 3) < 1e-6

    def test_success_rate_empty(self):
        from backend.chem.base import DescriptorResult
        dr = DescriptorResult(
            descriptors=pd.DataFrame(),
            smiles_list=[],
            failed_indices=[],
            adapter_name="test",
        )
        assert dr.success_rate == 0.0

    def test_n_descriptors(self):
        from backend.chem.base import DescriptorResult
        dr = DescriptorResult(
            descriptors=pd.DataFrame({"a": [1], "b": [2], "c": [3]}),
            smiles_list=["C"],
            failed_indices=[],
            adapter_name="test",
        )
        assert dr.n_descriptors == 3


class TestDescriptorMetadata:

    def test_creation(self):
        from backend.chem.base import DescriptorMetadata
        dm = DescriptorMetadata(
            name="MolWeight",
            meaning="分子量",
            is_count=False,
        )
        assert dm.name == "MolWeight"
        assert dm.is_binary is False


class TestBaseChemAdapter:

    def test_concrete_subclass(self):
        from backend.chem.base import BaseChemAdapter, DescriptorResult

        class DummyAdapter(BaseChemAdapter):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "ダミーアダプタ"

            def is_available(self) -> bool:
                return True

            def compute(self, smiles_list, **kwargs):
                df = pd.DataFrame({"dummy_feat": range(len(smiles_list))})
                return DescriptorResult(
                    descriptors=df,
                    smiles_list=smiles_list,
                    failed_indices=[],
                    adapter_name=self.name,
                )

        adapter = DummyAdapter()
        assert adapter.name == "dummy"
        assert adapter.is_available() is True
        result = adapter.compute(["C", "CC"])
        assert result.n_descriptors == 1
        assert result.success_rate == 1.0

    def test_require_available_raises(self):
        from backend.chem.base import BaseChemAdapter, DescriptorResult

        class UnavailableAdapter(BaseChemAdapter):
            @property
            def name(self) -> str:
                return "unavailable"

            @property
            def description(self) -> str:
                return "未インストール"

            def is_available(self) -> bool:
                return False

            def compute(self, smiles_list, **kwargs):
                self._require_available()

        adapter = UnavailableAdapter()
        with pytest.raises(RuntimeError, match="インストールされていません"):
            adapter.compute(["C"])

    def test_repr(self):
        from backend.chem.base import BaseChemAdapter, DescriptorResult

        class RepAdapter(BaseChemAdapter):
            @property
            def name(self) -> str:
                return "rep"

            @property
            def description(self) -> str:
                return "repr test"

            def is_available(self) -> bool:
                return True

            def compute(self, smiles_list, **kwargs):
                pass

        adapter = RepAdapter()
        r = repr(adapter)
        assert "rep" in r
        assert "available" in r

    def test_get_descriptor_names_default(self):
        from backend.chem.base import BaseChemAdapter, DescriptorResult

        class MinimalAdapter(BaseChemAdapter):
            @property
            def name(self) -> str:
                return "minimal"

            @property
            def description(self) -> str:
                return "minimal"

            def is_available(self) -> bool:
                return True

            def compute(self, smiles_list, **kwargs):
                pass

        adapter = MinimalAdapter()
        assert adapter.get_descriptor_names() == []
        assert adapter.get_descriptors_metadata() == []


# ═══════════════════════════════════════════════════════════════════
# benchmark_datasets.py テスト
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkDatasets:

    def test_list_benchmark_datasets(self):
        from backend.data.benchmark_datasets import list_benchmark_datasets
        datasets = list_benchmark_datasets()
        assert len(datasets) == 3
        ids = [d["id"] for d in datasets]
        assert "esol" in ids
        assert "freesolv" in ids
        assert "lipophilicity" in ids

    def test_list_has_required_fields(self):
        from backend.data.benchmark_datasets import list_benchmark_datasets
        for ds in list_benchmark_datasets():
            assert "id" in ds
            assert "name" in ds
            assert "description" in ds
            assert "target" in ds

    def test_load_benchmark_invalid(self):
        from backend.data.benchmark_datasets import load_benchmark
        with pytest.raises(ValueError, match="未知"):
            load_benchmark("nonexistent_dataset")

    def test_benchmark_urls_exist(self):
        from backend.data.benchmark_datasets import BENCHMARK_URLS
        assert "esol" in BENCHMARK_URLS
        assert all(url.startswith("https://") for url in BENCHMARK_URLS.values())


# ═══════════════════════════════════════════════════════════════════
# molai_adapter.py テスト
# ═══════════════════════════════════════════════════════════════════

class TestMolAITokenizer:

    def test_tokenize_simple(self):
        from backend.chem.molai_adapter import _tokenize_smiles
        tokens = _tokenize_smiles("CCO")
        assert tokens == ["C", "C", "O"]

    def test_tokenize_br_cl(self):
        from backend.chem.molai_adapter import _tokenize_smiles
        tokens = _tokenize_smiles("CBr")
        assert "Br" in tokens

        tokens2 = _tokenize_smiles("CCl")
        assert "Cl" in tokens2

    def test_smiles_to_onehot_shape(self):
        from backend.chem.molai_adapter import _smiles_to_onehot, MAX_SMILES_LEN, VOCAB_SIZE
        oh = _smiles_to_onehot("CCO")
        assert oh.shape == (MAX_SMILES_LEN, VOCAB_SIZE)
        # 各行はone-hot（合計1.0）
        assert np.allclose(oh.sum(axis=1), 1.0)


class TestMolAIAdapter:

    def test_name_and_description(self):
        from backend.chem.molai_adapter import MolAIAdapter
        adapter = MolAIAdapter()
        assert adapter.name == "molai"
        assert len(adapter.description) > 0

    def test_get_descriptor_names(self):
        from backend.chem.molai_adapter import MolAIAdapter
        adapter = MolAIAdapter(n_components=5)
        names = adapter.get_descriptor_names()
        assert len(names) == 5
        assert names[0] == "molai_pc1"

    def test_get_descriptors_metadata(self):
        from backend.chem.molai_adapter import MolAIAdapter
        adapter = MolAIAdapter(n_components=3)
        meta = adapter.get_descriptors_metadata()
        assert len(meta) == 3
        assert meta[0].name == "molai_pc1"
        assert meta[0].is_count is False

    def test_compute_if_torch_available(self):
        from backend.chem.molai_adapter import MolAIAdapter
        adapter = MolAIAdapter(n_components=4, latent_dim=64)
        if not adapter.is_available():
            pytest.skip("torch未インストール")
        result = adapter.compute(["CCO", "c1ccccc1", "CC(=O)O"])
        assert result.descriptors.shape[0] == 3
        assert result.descriptors.shape[1] > 0
        assert result.descriptors.shape[1] <= 4
        assert result.adapter_name == "molai"
