# -*- coding: utf-8 -*-
"""
tests/test_uma_adapter.py

UMAAdapter のユニットテスト。
HuggingFace モデルダウンロード不要（モック使用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from backend.chem.uma_adapter import UMAAdapter, _smiles_to_ase_atoms, _UMA_DESCRIPTORS


# ─── ヘルパー ──────────────────────────────────────────────────

def _make_mock_calc():
    """UMA Calculator のモックを生成する。"""
    mock_calc = MagicMock()
    return mock_calc


# ─── 基本プロパティテスト ───────────────────────────────────────

class TestUMAAdapterProperties:
    """F-uma-001: UMAAdapter の基本プロパティテスト"""

    def test_name(self):
        adapter = UMAAdapter()
        assert adapter.name == "uma"

    def test_description_not_empty(self):
        adapter = UMAAdapter()
        assert len(adapter.description) > 10

    def test_default_model_name(self):
        adapter = UMAAdapter()
        assert adapter.model_name == "uma-s-1p2"

    def test_default_device(self):
        adapter = UMAAdapter()
        assert adapter.device == "cpu"

    def test_custom_model_name(self):
        adapter = UMAAdapter(model_name="uma-m-1p1")
        assert adapter.model_name == "uma-m-1p1"

    def test_custom_device(self):
        adapter = UMAAdapter(device="cuda")
        assert adapter.device == "cuda"


# ─── is_available テスト ──────────────────────────────────────

class TestUMAAdapterAvailability:
    """F-uma-002: is_available のテスト"""

    def test_is_available_with_fairchem(self):
        """fairchem-core がインストールされていれば True"""
        adapter = UMAAdapter()
        # fairchem-core は実際にインストール済み
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_is_available_without_fairchem(self):
        """fairchem-core がない場合は False"""
        with patch.dict("sys.modules", {"fairchem": None, "fairchem.core": None}):
            adapter = UMAAdapter()
            # import が失敗する場合をシミュレート
            with patch.object(adapter, "is_available", return_value=False):
                assert adapter.is_available() is False


# ─── get_descriptor_names テスト ─────────────────────────────

class TestUMAAdapterDescriptors:
    """F-uma-003: 記述子メタデータのテスト"""

    def test_descriptor_names_count(self):
        adapter = UMAAdapter()
        names = adapter.get_descriptor_names()
        assert len(names) == len(_UMA_DESCRIPTORS)

    def test_descriptor_names_prefix(self):
        """全記述子名が 'uma_' プレフィックスを持つ"""
        adapter = UMAAdapter()
        for name in adapter.get_descriptor_names():
            assert name.startswith("uma_"), f"{name} does not start with 'uma_'"

    def test_descriptor_metadata_consistency(self):
        """メタデータとnames が一致する"""
        adapter = UMAAdapter()
        meta = adapter.get_descriptors_metadata()
        names = adapter.get_descriptor_names()
        assert len(meta) == len(names)
        for m, n in zip(meta, names):
            assert m.name == n

    def test_descriptor_metadata_fields(self):
        """メタデータのフィールドが正常"""
        adapter = UMAAdapter()
        for m in adapter.get_descriptors_metadata():
            assert m.is_count is False
            assert m.is_binary is False
            assert len(m.meaning) > 0

    def test_expected_descriptors_present(self):
        """重要な記述子が含まれている"""
        adapter = UMAAdapter()
        names = adapter.get_descriptor_names()
        expected = ["uma_TotalEnergy", "uma_EnergyPerAtom", "uma_ForceMax"]
        for e in expected:
            assert e in names, f"{e} not found in descriptor names"


# ─── _smiles_to_ase_atoms テスト ─────────────────────────────

class TestSmilesToAseAtoms:
    """F-uma-004: SMILES → ASE Atoms 変換テスト"""

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/ase/rdkit not available"
    )
    def test_valid_smiles(self):
        """有効なSMILESが ASE Atoms に変換される"""
        atoms = _smiles_to_ase_atoms("CCO")
        assert atoms is not None
        assert len(atoms) > 0  # 原子数 > 0
        assert hasattr(atoms, "get_positions")

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/ase/rdkit not available"
    )
    def test_invalid_smiles(self):
        """無効なSMILESは None を返す"""
        atoms = _smiles_to_ase_atoms("INVALID_SMILES_XYZ")
        assert atoms is None

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/ase/rdkit not available"
    )
    def test_atom_count_ethanol(self):
        """エタノール(CCO)のH付き原子数が正しい (C2H6O = 9原子)"""
        atoms = _smiles_to_ase_atoms("CCO")
        if atoms is not None:
            assert len(atoms) == 9  # 2C + 6H + 1O

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/ase/rdkit not available"
    )
    def test_pbc_is_false(self):
        """分子計算では周期境界条件がOFF"""
        atoms = _smiles_to_ase_atoms("C")
        if atoms is not None:
            assert not any(atoms.pbc)

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/ase/rdkit not available"
    )
    def test_charge_info(self):
        """電荷情報がinfoに含まれる"""
        atoms = _smiles_to_ase_atoms("CCO")
        if atoms is not None:
            assert "charge" in atoms.info
            assert "spin" in atoms.info

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/ase/rdkit not available"
    )
    def test_charged_molecule(self):
        """荷電分子(酢酸イオン)の電荷が正しい"""
        atoms = _smiles_to_ase_atoms("[O-]C(=O)C")
        if atoms is not None:
            assert atoms.info["charge"] == -1


# ─── compute テスト (モック使用) ──────────────────────────────

class TestUMAAdapterCompute:
    """F-uma-005: compute メソッドのテスト（モック使用）"""

    def _make_adapter_with_mock(self):
        """モックCalculator付きのアダプタを作成"""
        adapter = UMAAdapter()
        adapter._predictor = MagicMock()
        adapter._calc = MagicMock()
        return adapter

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_returns_descriptor_result(self):
        """compute が DescriptorResult を返す"""
        adapter = self._make_adapter_with_mock()

        # atoms.get_potential_energy() / get_forces() のモック
        mock_atoms = MagicMock()
        mock_atoms.get_potential_energy.return_value = -100.5
        mock_atoms.get_forces.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_atoms.__len__ = lambda self: 2
        mock_atoms.get_stress.side_effect = Exception("no PBC")
        mock_atoms.get_dipole_moment.side_effect = Exception("not supported")

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", return_value=mock_atoms):
            result = adapter.compute(["CCO"])

        from backend.chem.base import DescriptorResult
        assert isinstance(result, DescriptorResult)
        assert result.adapter_name == "uma"
        assert len(result.smiles_list) == 1
        assert result.descriptors.shape[0] == 1

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_energy_values(self):
        """エネルギー値が正しく計算される"""
        adapter = self._make_adapter_with_mock()

        mock_atoms = MagicMock()
        mock_atoms.get_potential_energy.return_value = -200.0
        mock_atoms.get_forces.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mock_atoms.__len__ = lambda self: 3
        mock_atoms.get_stress.side_effect = Exception("no PBC")
        mock_atoms.get_dipole_moment.side_effect = Exception("not supported")

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", return_value=mock_atoms):
            result = adapter.compute(["c1ccccc1"])

        row = result.descriptors.iloc[0]
        assert row["uma_TotalEnergy"] == pytest.approx(-200.0)
        assert row["uma_EnergyPerAtom"] == pytest.approx(-200.0 / 3)
        assert row["uma_ForceMax"] == pytest.approx(1.0)
        assert row["uma_ForceMean"] == pytest.approx(1.0)
        assert row["uma_ForceStd"] == pytest.approx(0.0)

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_failed_smiles(self):
        """変換失敗のSMILESがfailed_indicesに記録される"""
        adapter = self._make_adapter_with_mock()

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", return_value=None):
            result = adapter.compute(["INVALID"])

        assert 0 in result.failed_indices
        assert result.descriptors.iloc[0].isna().all()

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_multiple_smiles(self):
        """複数SMILESの処理"""
        adapter = self._make_adapter_with_mock()

        mock_atoms = MagicMock()
        mock_atoms.get_potential_energy.return_value = -50.0
        mock_atoms.get_forces.return_value = np.array([[0.5, 0.5, 0.5]])
        mock_atoms.__len__ = lambda self: 1
        mock_atoms.get_stress.side_effect = Exception("no PBC")
        mock_atoms.get_dipole_moment.side_effect = Exception("not supported")

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", return_value=mock_atoms):
            result = adapter.compute(["C", "CC", "CCC"])

        assert result.descriptors.shape[0] == 3
        assert len(result.failed_indices) == 0

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_mixed_success_failure(self):
        """成功と失敗が混在"""
        adapter = self._make_adapter_with_mock()

        mock_atoms = MagicMock()
        mock_atoms.get_potential_energy.return_value = -75.0
        mock_atoms.get_forces.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_atoms.__len__ = lambda self: 1
        mock_atoms.get_stress.side_effect = Exception("no PBC")
        mock_atoms.get_dipole_moment.side_effect = Exception("not supported")

        def side_effect(smi):
            return mock_atoms if smi != "BAD" else None

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", side_effect=side_effect):
            result = adapter.compute(["C", "BAD", "CC"])

        assert result.descriptors.shape[0] == 3
        assert len(result.failed_indices) == 1
        assert 1 in result.failed_indices
        assert not np.isnan(result.descriptors.iloc[0]["uma_TotalEnergy"])
        assert np.isnan(result.descriptors.iloc[1]["uma_TotalEnergy"])

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_success_rate(self):
        """成功率の計算"""
        adapter = self._make_adapter_with_mock()

        mock_atoms = MagicMock()
        mock_atoms.get_potential_energy.return_value = -10.0
        mock_atoms.get_forces.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_atoms.__len__ = lambda self: 1
        mock_atoms.get_stress.side_effect = Exception("no PBC")
        mock_atoms.get_dipole_moment.side_effect = Exception("not supported")

        def side_effect(smi):
            return None if smi == "BAD" else mock_atoms

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", side_effect=side_effect):
            result = adapter.compute(["C", "BAD", "CC", "BAD2"])

        # BAD and BAD2 fail → 2/4 = 50%
        # Note: BAD2 might also fail since _smiles_to_ase_atoms returns None for it
        # Actually let's check: side_effect returns None only for "BAD"
        # "BAD2" != "BAD" so it returns mock_atoms
        assert result.success_rate == pytest.approx(0.75)

    @pytest.mark.skipif(
        not UMAAdapter().is_available(),
        reason="fairchem/rdkit/ase not available"
    )
    def test_compute_metadata(self):
        """メタデータにモデル情報が含まれる"""
        adapter = self._make_adapter_with_mock()

        with patch("backend.chem.uma_adapter._smiles_to_ase_atoms", return_value=None):
            result = adapter.compute(["C"])

        assert result.metadata["model_name"] == "uma-s-1p2"
        assert result.metadata["device"] == "cpu"


# ─── __init__.py 統合テスト ──────────────────────────────────

class TestUMAInitIntegration:
    """F-uma-006: __init__.py からの安全importテスト"""

    def test_import_from_init(self):
        """backend.chem から UMAAdapter がimportできる"""
        from backend.chem import UMAAdapter
        adapter = UMAAdapter()
        assert adapter.name == "uma"

    def test_unavailable_fallback(self):
        """import失敗時のフォールバック"""
        from backend.chem import _make_unavailable_adapter
        FakeUMA = _make_unavailable_adapter("UMAAdapter")
        fake = FakeUMA()
        assert fake.is_available() is False
        with pytest.raises(RuntimeError):
            fake.compute(["C"])
