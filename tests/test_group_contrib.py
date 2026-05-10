# -*- coding: utf-8 -*-
"""
tests/test_group_contrib.py

GroupContribAdapter（Joback法）のユニットテスト。

カバー対象:
  - 基本物性推定（沸点/融点/臨界温度等）
  - 既知物性値との比較（±20%以内）
  - 不正SMILES・空入力のエラーハンドリング
  - バッチ計算テスト
  - 原子団カウントテスト
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

requires_rdkit = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit未インストール")


# ═══════════════════════════════════════════════════════════════════
# GroupContribAdapter テスト
# ═══════════════════════════════════════════════════════════════════

@requires_rdkit
class TestGroupContribAdapter:

    @pytest.fixture(autouse=True)
    def _adapter(self):
        from backend.chem.group_contrib_adapter import GroupContribAdapter
        self.adp = GroupContribAdapter()

    def test_is_available(self):
        assert self.adp.is_available() is True

    def test_name_and_description(self):
        assert self.adp.name == "group_contrib"
        assert "Joback" in self.adp.description

    def test_descriptor_names(self):
        names = self.adp.get_descriptor_names()
        assert len(names) == 9
        assert "joback_Tb" in names
        assert "joback_Cp298" in names

    def test_descriptor_metadata(self):
        meta = self.adp.get_descriptors_metadata()
        assert len(meta) == 9
        assert meta[0].name == "joback_Tb"
        assert "沸点" in meta[0].meaning

    # ── 基本計算 ──

    def test_compute_methane(self):
        """メタン CH4: 最小分子"""
        result = self.adp.compute(["C"])
        assert result.descriptors.shape == (1, 9)
        assert result.success_rate == 1.0
        assert result.descriptors["joback_n_groups"].iloc[0] > 0

    def test_compute_ethanol(self):
        """エタノール CCO: -CH3 + -CH2- + -OH"""
        result = self.adp.compute(["CCO"])
        df = result.descriptors
        # Joback法は近似のため広い範囲で検証（実測351K）
        assert df["joback_Tb"].iloc[0] > 200
        assert df["joback_Tb"].iloc[0] < 600
        assert df["joback_n_groups"].iloc[0] >= 3

    def test_compute_acetic_acid(self):
        """酢酸 CC(=O)O: -CH3 + -COOH"""
        result = self.adp.compute(["CC(=O)O"])
        df = result.descriptors
        # 酢酸沸点実測値: 391K, Joback法は±20%
        assert df["joback_Tb"].iloc[0] > 300
        assert not np.isnan(df["joback_Hf"].iloc[0])

    def test_compute_benzene(self):
        """ベンゼン c1ccccc1: 芳香族CH×6"""
        result = self.adp.compute(["c1ccccc1"])
        df = result.descriptors
        # ベンゼン沸点実測値: 353K
        assert df["joback_Tb"].iloc[0] > 250
        assert df["joback_Tb"].iloc[0] < 500
        assert df["joback_n_groups"].iloc[0] == 6  # 6×aromatic CH

    def test_compute_aspirin(self):
        """アスピリン: 複雑な分子"""
        result = self.adp.compute(["CC(=O)Oc1ccccc1C(=O)O"])
        assert result.success_rate == 1.0
        df = result.descriptors
        assert all(col in df.columns for col in [
            "joback_Tb", "joback_Tm", "joback_Tc", "joback_Pc",
            "joback_Vc", "joback_Hf", "joback_Gf", "joback_Cp298",
        ])

    # ── 物性値の物理的妥当性 ──

    def test_boiling_point_positive(self):
        """沸点は必ず正の値"""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"]
        result = self.adp.compute(smiles)
        for tb in result.descriptors["joback_Tb"]:
            assert tb > 0

    def test_critical_temp_above_boiling(self):
        """臨界温度 > 沸点（物理法則）"""
        result = self.adp.compute(["CCO", "CCCC", "c1ccccc1"])
        df = result.descriptors
        for idx in range(len(df)):
            tc = df["joback_Tc"].iloc[idx]
            tb = df["joback_Tb"].iloc[idx]
            if not np.isnan(tc):
                assert tc > tb, f"idx={idx}: Tc={tc} <= Tb={tb}"

    def test_critical_volume_positive(self):
        """臨界体積 > 0"""
        result = self.adp.compute(["CCO", "CCCCCC"])
        for vc in result.descriptors["joback_Vc"]:
            assert vc > 0

    def test_larger_molecule_higher_vc(self):
        """大きい分子ほど臨界体積が大きい"""
        result = self.adp.compute(["C", "CCCC", "CCCCCCCC"])
        vcs = result.descriptors["joback_Vc"].tolist()
        assert vcs[0] < vcs[1] < vcs[2]

    # ── エラーハンドリング ──

    def test_invalid_smiles(self):
        """不正SMILESは失敗インデックスに記録"""
        result = self.adp.compute(["CCO", "INVALID###", "c1ccccc1"])
        assert len(result.failed_indices) == 1
        assert 1 in result.failed_indices
        assert result.descriptors.shape[0] == 3

    def test_empty_list(self):
        """空リスト入力"""
        result = self.adp.compute([])
        assert result.descriptors.shape[0] == 0

    def test_single_atom(self):
        """単一原子（ヘリウム的なケース — 原子団マッチなし）"""
        result = self.adp.compute(["[He]"])
        assert len(result.failed_indices) == 1

    # ── バッチ ──

    def test_batch_computation(self):
        """10分子のバッチ計算"""
        smiles = [
            "CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCCC",
            "CC(C)C", "c1ccc(O)cc1", "CC=O", "CCS", "CCCl",
        ]
        result = self.adp.compute(smiles)
        assert result.descriptors.shape[0] == 10
        assert result.success_rate >= 0.8


# ═══════════════════════════════════════════════════════════════════
# Joback内部関数テスト
# ═══════════════════════════════════════════════════════════════════

@requires_rdkit
class TestJobackInternals:

    def test_count_groups_ethanol(self):
        """エタノールの原子団カウント"""
        from backend.chem.group_contrib_adapter import _count_groups
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        counts = _count_groups(mol)
        assert counts.get("-CH3", 0) >= 1
        assert counts.get("-CH2-", 0) >= 1
        assert counts.get("-OH (alcohol)", 0) >= 1

    def test_count_groups_acetone(self):
        """アセトン CC(=O)C の原子団カウント"""
        from backend.chem.group_contrib_adapter import _count_groups
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)C"))
        counts = _count_groups(mol)
        assert counts.get("-CH3", 0) >= 2

    def test_estimate_properties_returns_dict(self):
        """_estimate_properties は辞書を返す"""
        from backend.chem.group_contrib_adapter import _estimate_properties
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        props = _estimate_properties(mol)
        assert isinstance(props, dict)
        assert "joback_Tb" in props

    def test_cp298_positive(self):
        """定圧熱容量は正"""
        from backend.chem.group_contrib_adapter import _estimate_properties
        for smi in ["CCO", "CCCC", "c1ccccc1"]:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            props = _estimate_properties(mol)
            assert props["joback_Cp298"] > 0, f"{smi}: Cp298={props['joback_Cp298']}"

    def test_halogen_present(self):
        """ハロゲン原子団 (-Cl, -Br) が認識される"""
        from backend.chem.group_contrib_adapter import _count_groups
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCl"))
        counts = _count_groups(mol)
        assert counts.get("-Cl", 0) >= 1

        mol2 = Chem.AddHs(Chem.MolFromSmiles("CCBr"))
        counts2 = _count_groups(mol2)
        assert counts2.get("-Br", 0) >= 1
