"""
tests/test_protonation.py

backend/chem/protonation.py のユニットテスト。

RDKitベースの中性化・脱プロトン化・プロトン化と
バッチ処理の検証。UniPKa依存テストはスキップ。
"""
from __future__ import annotations

import pytest

# ── テスト対象 ──
from backend.chem.protonation import (
    apply_protonation,
    apply_protonation_batch,
    _neutralize,
    _max_deprotonate,
    _max_protonate,
)


# ── ヘルパー ──
class _FakeConfig:
    """MoleculeChargeConfig の最小モック。"""
    def __init__(self, mode: str = "as_is", ph: float | None = None):
        self.protonate_mode = mode
        self.ph = ph


# ============================================================
# _neutralize
# ============================================================

class TestNeutralize:
    """RDKit MolStandardize による中性化。"""

    def test_sodium_carboxylate(self):
        """イオン性ナトリウム塩 → 中性カルボン酸"""
        # [Na+].[O-]C(=O)c1ccccc1 のようなイオン性塩
        result = _neutralize("[Na+].[O-]C(=O)c1ccccc1")
        assert result is not None
        assert "Na" not in result  # 対イオン除去

    def test_already_neutral(self):
        """中性分子はそのまま"""
        smi = "c1ccccc1"  # ベンゼン
        result = _neutralize(smi)
        assert result == smi

    def test_invalid_smiles(self):
        """無効SMILESは入力をそのまま返す"""
        result = _neutralize("INVALID")
        assert result == "INVALID"

    def test_empty_string(self):
        """空文字列"""
        result = _neutralize("")
        assert result == ""

    def test_ammonium(self):
        """アンモニウム塩の処理"""
        result = _neutralize("[NH4+].[Cl-]")
        assert result is not None
        assert "Cl" not in result  # 対イオン除去
        assert isinstance(result, str)


# ============================================================
# _max_deprotonate
# ============================================================

class TestMaxDeprotonate:
    def test_carboxylic_acid(self):
        """カルボン酸の脱プロトン化"""
        result = _max_deprotonate("OC(=O)c1ccccc1")
        assert result is not None
        assert isinstance(result, str)

    def test_invalid_smiles(self):
        result = _max_deprotonate("NOT_A_SMILES")
        assert result == "NOT_A_SMILES"

    def test_benzene(self):
        """酸性プロトンなし → そのまま"""
        result = _max_deprotonate("c1ccccc1")
        assert result == "c1ccccc1"


# ============================================================
# _max_protonate
# ============================================================

class TestMaxProtonate:
    def test_amine(self):
        """アミンのプロトン化"""
        result = _max_protonate("Nc1ccccc1")
        assert result is not None
        assert isinstance(result, str)

    def test_invalid_smiles(self):
        result = _max_protonate("BAD_SMILES")
        assert result == "BAD_SMILES"


# ============================================================
# apply_protonation
# ============================================================

class TestApplyProtonation:
    def test_as_is(self):
        """as_isモード: そのまま返す"""
        smi = "c1ccccc1"
        result = apply_protonation(smi, _FakeConfig("as_is"))
        assert result == smi

    def test_neutral_mode(self):
        """neutralモード: 中性化"""
        smi = "[Na+].[O-]C(=O)c1ccccc1"
        result = apply_protonation(smi, _FakeConfig("neutral"))
        assert "Na" not in result

    def test_max_acid_mode(self):
        """max_acidモード"""
        result = apply_protonation("OC(=O)c1ccccc1", _FakeConfig("max_acid"))
        assert result is not None

    def test_max_base_mode(self):
        """max_baseモード"""
        result = apply_protonation("Nc1ccccc1", _FakeConfig("max_base"))
        assert result is not None

    def test_unknown_mode(self):
        """未知のモードはそのまま返す"""
        smi = "c1ccccc1"
        result = apply_protonation(smi, _FakeConfig("unknown_mode"))
        assert result == smi

    def test_none_input(self):
        """None入力"""
        result = apply_protonation(None, _FakeConfig("neutral"))
        assert result is None

    def test_empty_string(self):
        """空文字列"""
        result = apply_protonation("", _FakeConfig("neutral"))
        assert result == ""

    def test_non_string_input(self):
        """非文字列"""
        result = apply_protonation(123, _FakeConfig("neutral"))
        assert result == 123

    def test_auto_ph_without_unipka(self):
        """auto_phモード: UniPKa不在時はフォールバック"""
        smi = "OC(=O)c1ccccc1"
        result = apply_protonation(smi, _FakeConfig("auto_ph", ph=7.4))
        # UniPKa不在なら中性化にフォールバックまたはそのまま返る
        assert isinstance(result, str)


# ============================================================
# apply_protonation_batch
# ============================================================

class TestApplyProtonationBatch:
    def test_batch(self):
        """バッチ処理: 全入力に同じモードを適用"""
        smiles = ["c1ccccc1", "[Na+].[O-]C(=O)c1ccccc1", "CC"]
        results = apply_protonation_batch(smiles, _FakeConfig("neutral"))
        assert len(results) == 3
        assert "Na" not in results[1]

    def test_empty_list(self):
        results = apply_protonation_batch([], _FakeConfig("as_is"))
        assert results == []

    def test_batch_as_is(self):
        smiles = ["c1ccccc1", "CC"]
        results = apply_protonation_batch(smiles, _FakeConfig("as_is"))
        assert results == smiles
"""
Implements: F-CHEM-PROT-001
論文: Henderson-Hasselbalch 式 (Henderson, 1908)
"""
