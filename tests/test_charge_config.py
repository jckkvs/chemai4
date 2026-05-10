# -*- coding: utf-8 -*-
"""
tests/test_charge_config.py

backend/chem/charge_config.py と backend/chem/protonation.py の
ユニットテスト（エッジケースまで網羅）。

カバー対象:
  - MoleculeChargeConfig: バリデーション・プロパティ・プリセット
  - ChargeConfigStore: 優先順位・ロジック
  - _read_smiles_formal_charge: 正常・異常・エッジケース
  - protonation.py: as_is / neutral / auto_ph / max_acid / max_base
  - PermissionError（WinError 32）フォールバック
"""
from __future__ import annotations

import threading
import unittest.mock as mock

import pytest

from backend.chem.charge_config import (
    ChargeConfigStore,
    MoleculeChargeConfig,
    _read_smiles_formal_charge,
)

# RDKit可用チェック
try:
    from rdkit import Chem as _rdkit_Chem  # noqa: F401
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

requires_rdkit = pytest.mark.skipif(
    not RDKIT_AVAILABLE,
    reason="RDKitがインストールされていないためスキップ",
)


# ===================================================================
# フィクスチャ
# ===================================================================

@pytest.fixture
def default_cfg() -> MoleculeChargeConfig:
    return MoleculeChargeConfig.default()


@pytest.fixture
def store() -> ChargeConfigStore:
    return ChargeConfigStore()


# ===================================================================
# MoleculeChargeConfig: デフォルト・プロパティ
# ===================================================================

class TestMoleculeChargeConfigDefaults:
    def test_default_values(self, default_cfg):
        assert default_cfg.formal_charge == 0
        assert default_cfg.spin_multiplicity == 1
        assert default_cfg.ph is None
        assert default_cfg.protonate_mode == "as_is"
        assert default_cfg.partial_charge_model == "gasteiger"
        assert default_cfg.consider_tautomers is False
        assert default_cfg.auto_charge_from_smiles is True

    def test_uhf_property_closed_shell(self, default_cfg):
        """閉殻: uhf=0"""
        assert default_cfg.uhf == 0

    def test_uhf_property_radical(self):
        cfg = MoleculeChargeConfig(spin_multiplicity=2)
        assert cfg.uhf == 1

    def test_uhf_property_triplet(self):
        cfg = MoleculeChargeConfig(spin_multiplicity=3)
        assert cfg.uhf == 2

    def test_to_xtb_args_neutral_closed_shell(self, default_cfg):
        """中性・閉殻のXTB引数"""
        args = default_cfg.to_xtb_args()
        assert "--chrg" in args
        assert "0" in args
        assert "--uhf" not in args  # 閉殻のとき --uhf は渡さない

    def test_to_xtb_args_cation_radical(self):
        """+1カチオン・ラジカルのXTB引数"""
        cfg = MoleculeChargeConfig(
            formal_charge=1, spin_multiplicity=2, auto_charge_from_smiles=False
        )
        args = cfg.to_xtb_args()
        assert args == ["--chrg", "1", "--uhf", "1"]

    def test_to_xtb_args_charge_override(self, default_cfg):
        """charge_override が formal_charge より優先されること"""
        args = default_cfg.to_xtb_args(charge_override=-2)
        assert args[0] == "--chrg"
        assert args[1] == "-2"

    def test_to_xtb_args_anion(self):
        cfg = MoleculeChargeConfig(formal_charge=-2, auto_charge_from_smiles=False)
        args = cfg.to_xtb_args()
        assert args[:2] == ["--chrg", "-2"]
        assert "--uhf" not in args


# ===================================================================
# MoleculeChargeConfig: バリデーション
# ===================================================================

class TestMoleculeChargeConfigValidation:
    def test_spin_multiplicity_zero_raises(self):
        with pytest.raises(ValueError, match="spin_multiplicity"):
            MoleculeChargeConfig(spin_multiplicity=0)

    def test_spin_multiplicity_negative_raises(self):
        with pytest.raises(ValueError):
            MoleculeChargeConfig(spin_multiplicity=-1)

    def test_formal_charge_too_large_raises(self):
        with pytest.raises(ValueError, match="formal_charge"):
            MoleculeChargeConfig(formal_charge=11)

    def test_formal_charge_too_negative_raises(self):
        with pytest.raises(ValueError, match="formal_charge"):
            MoleculeChargeConfig(formal_charge=-11)

    def test_boundary_charge_plus_10(self):
        cfg = MoleculeChargeConfig(formal_charge=10)
        assert cfg.formal_charge == 10

    def test_boundary_charge_minus_10(self):
        cfg = MoleculeChargeConfig(formal_charge=-10)
        assert cfg.formal_charge == -10

    def test_spin_multiplicity_4(self):
        """四重項（ラジカル）"""
        cfg = MoleculeChargeConfig(spin_multiplicity=4)
        assert cfg.uhf == 3


# ===================================================================
# MoleculeChargeConfig: プリセット
# ===================================================================

class TestMoleculeChargeConfigPresets:
    def test_for_radical_default_neutral(self):
        cfg = MoleculeChargeConfig.for_radical()
        assert cfg.spin_multiplicity == 2
        assert cfg.formal_charge == 0

    def test_for_radical_charged(self):
        cfg = MoleculeChargeConfig.for_radical(charge=+1)
        assert cfg.formal_charge == 1
        assert cfg.spin_multiplicity == 2

    def test_at_physiological_ph(self):
        cfg = MoleculeChargeConfig.at_physiological_ph()
        assert cfg.protonate_mode == "auto_ph"
        assert cfg.ph == pytest.approx(7.4)
        assert cfg.partial_charge_model == "gasteiger"


# ===================================================================
# ChargeConfigStore: 優先順位とロジック
# ===================================================================

class TestChargeConfigStore:
    NEUTRAL = "CCO"
    CATION = "[NH4+]"
    ANION = "CC(=O)[O-]"
    DIANION = "[O-]S(=O)(=O)[O-]"

    def test_get_config_returns_default_for_unknown_smiles(self, store):
        cfg = store.get_config("CCCCC")
        assert cfg is store.default

    def test_set_and_get_per_molecule(self, store):
        custom = MoleculeChargeConfig(spin_multiplicity=2)
        store.set_per_molecule(self.CATION, custom)
        assert store.get_config(self.CATION) is custom

    def test_per_molecule_overrides_default(self, store):
        store.default.formal_charge = 0
        custom = MoleculeChargeConfig(
            formal_charge=-1, auto_charge_from_smiles=False
        )
        store.set_per_molecule(self.ANION, custom)
        cfg = store.get_config(self.ANION)
        assert cfg.formal_charge == -1

    @requires_rdkit
    def test_resolve_charge_from_smiles_cation(self, store):
        """[NH4+] は +1 をSMILESから自動検出"""
        store.default.auto_charge_from_smiles = True
        charge = store.resolve_charge(self.CATION)
        assert charge == 1

    @requires_rdkit
    def test_resolve_charge_from_smiles_anion(self, store):
        """CC(=O)[O-] は -1 をSMILESから自動検出"""
        store.default.auto_charge_from_smiles = True
        charge = store.resolve_charge(self.ANION)
        assert charge == -1

    @requires_rdkit
    def test_resolve_charge_from_smiles_dianion(self, store):
        """二価アニオン は -2"""
        store.default.auto_charge_from_smiles = True
        charge = store.resolve_charge(self.DIANION)
        assert charge == -2

    def test_resolve_charge_manual_override(self, store):
        """auto_charge_from_smiles=False のとき手動設定が使われる"""
        store.default.auto_charge_from_smiles = False
        store.default.formal_charge = 3
        charge = store.resolve_charge(self.NEUTRAL)
        assert charge == 3

    def test_resolve_spin_default(self, store):
        assert store.resolve_spin("CCO") == 1

    def test_resolve_spin_per_molecule(self, store):
        store.set_per_molecule("CCO", MoleculeChargeConfig(spin_multiplicity=2))
        assert store.resolve_spin("CCO") == 2

    def test_per_molecule_does_not_affect_others(self, store):
        store.set_per_molecule("CCO", MoleculeChargeConfig(spin_multiplicity=3))
        assert store.resolve_spin("CCC") == 1  # デフォルト値


# ===================================================================
# _read_smiles_formal_charge
# ===================================================================

class TestReadSmilesFormalCharge:
    @requires_rdkit
    def test_neutral_ethanol(self):
        assert _read_smiles_formal_charge("CCO") == 0

    @requires_rdkit
    def test_cation_ammonium(self):
        assert _read_smiles_formal_charge("[NH4+]") == 1

    @requires_rdkit
    def test_anion_acetate(self):
        assert _read_smiles_formal_charge("CC(=O)[O-]") == -1

    def test_invalid_smiles_returns_0(self):
        assert _read_smiles_formal_charge("INVALID###") == 0

    def test_empty_string_returns_0(self):
        assert _read_smiles_formal_charge("") == 0

    def test_none_returns_0(self):
        assert _read_smiles_formal_charge("None") == 0

    @requires_rdkit
    def test_zwitterion(self):
        """双性イオン（グリシン）全体の電荷=0"""
        charge = _read_smiles_formal_charge("[NH3+]CC(=O)[O-]")
        assert charge == 0

    @requires_rdkit
    def test_large_molecule_neutral(self):
        """大きな分子（アスピリン）"""
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        assert _read_smiles_formal_charge(aspirin) == 0


# ===================================================================
# Protonation: apply_protonation
# ===================================================================

class TestApplyProtonation:
    def setup_method(self):
        from backend.chem.protonation import apply_protonation
        self.fn = apply_protonation

    def test_as_is_returns_unchanged(self):
        cfg = MoleculeChargeConfig(protonate_mode="as_is")
        assert self.fn("[NH4+]", cfg) == "[NH4+]"

    def test_as_is_anion_unchanged(self):
        smi = "CC(=O)[O-]"
        cfg = MoleculeChargeConfig(protonate_mode="as_is")
        assert self.fn(smi, cfg) == smi

    @requires_rdkit
    def test_neutral_removes_charge_acetate(self):
        """酢酸アニオンを中性 CC(=O)O にする"""
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn("CC(=O)[O-]", cfg)
        assert result is not None
        from rdkit import Chem
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0

    @requires_rdkit
    def test_neutral_removes_charge_ammonium(self):
        """アンモニウムイオンを中性化"""
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn("[NH4+]", cfg)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0

    @requires_rdkit
    def test_neutral_already_neutral(self):
        """中性分子を中性化しても変わらない（またはそれに近い値）"""
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn("CCO", cfg)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(result)
        assert Chem.GetFormalCharge(mol) == 0

    @requires_rdkit
    def test_neutral_salt_desalted(self):
        """塩（Na酢酸）はフラグメント除去で有機分子のみ出力"""
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn("[Na+].CC(=O)[O-]", cfg)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0

    @requires_rdkit
    def test_auto_ph_without_unipka_falls_back_to_neutral(self):
        """UniPKa未利用の場合は中性にフォールバックする"""
        cfg = MoleculeChargeConfig(protonate_mode="auto_ph", ph=7.4)
        with mock.patch(
            "backend.chem.protonation._get_unipka_model",
            side_effect=ImportError("mock: unipka not installed"),
        ):
            result = self.fn("CC(=O)[O-]", cfg)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0

    @requires_rdkit
    def test_auto_ph_permission_error_falls_back_to_neutral(self):
        """WinError 32 (PermissionError) でも中性にフォールバック"""
        cfg = MoleculeChargeConfig(protonate_mode="auto_ph", ph=7.4)
        with mock.patch(
            "backend.chem.protonation._get_unipka_model",
            side_effect=PermissionError("[WinError 32] mock file lock"),
        ):
            result = self.fn("CC(=O)[O-]", cfg)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0

    def test_unknown_mode_returns_original(self):
        """未知のプロトン化モードはそのまま返す"""
        cfg = MoleculeChargeConfig(protonate_mode="as_is")
        object.__setattr__(cfg, "protonate_mode", "unknown_mode")
        result = self.fn("CCO", cfg)
        assert result == "CCO"

    def test_empty_string_returns_empty(self):
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn("", cfg)
        assert result == ""

    def test_none_input_returns_none(self):
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn(None, cfg)
        assert result is None

    def test_invalid_smiles_fallback(self):
        """不正SMILESは変換せずそのまま返す"""
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = self.fn("INVALID###", cfg)
        assert result == "INVALID###"


# ===================================================================
# Protonation: apply_protonation_batch
# ===================================================================

class TestApplyProtonationBatch:
    def test_batch_same_length(self):
        from backend.chem.protonation import apply_protonation_batch
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        smiles = ["CCO", "CC(=O)[O-]", "[NH4+]"]
        result = apply_protonation_batch(smiles, cfg)
        assert len(result) == 3

    def test_batch_as_is_identical(self):
        from backend.chem.protonation import apply_protonation_batch
        cfg = MoleculeChargeConfig(protonate_mode="as_is")
        smiles = ["CCO", "[NH4+]", "C1CCCCC1"]
        result = apply_protonation_batch(smiles, cfg)
        assert result == smiles

    def test_batch_with_invalid_smiles(self):
        """不正SMILESを含むバッチでもクラッシュしない"""
        from backend.chem.protonation import apply_protonation_batch
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        smiles = ["CCO", "INVALID###", "c1ccccc1"]
        result = apply_protonation_batch(smiles, cfg)
        assert len(result) == 3
        assert result[1] == "INVALID###"


# ===================================================================
# Protonation: get_protonation_state_info
# ===================================================================

class TestGetProtonationStateInfo:
    def test_returns_dict_keys(self):
        from backend.chem.protonation import get_protonation_state_info
        with mock.patch(
            "backend.chem.protonation._get_unipka_model",
            side_effect=ImportError("mock: unipka not installed"),
        ):
            info = get_protonation_state_info("CCO", ph=7.4)
        assert "pka_acidic" in info
        assert "pka_basic" in info
        assert "dominant_form_at_ph" in info
        assert "ionization_note" in info

    def test_import_error_message(self):
        from backend.chem.protonation import get_protonation_state_info
        with mock.patch(
            "backend.chem.protonation._get_unipka_model",
            side_effect=ImportError("mock"),
        ):
            info = get_protonation_state_info("CCO")
        assert "インストール" in info["ionization_note"]

    def test_permission_error_message(self):
        from backend.chem.protonation import get_protonation_state_info
        with mock.patch(
            "backend.chem.protonation._get_unipka_model",
            side_effect=PermissionError("[WinError 32] mock"),
        ):
            info = get_protonation_state_info("CCO")
        assert "ロック" in info["ionization_note"] or "プロセス" in info["ionization_note"]

    def test_generic_exception_shows_error(self):
        from backend.chem.protonation import get_protonation_state_info
        with mock.patch(
            "backend.chem.protonation._get_unipka_model",
            side_effect=RuntimeError("some unexpected error"),
        ):
            info = get_protonation_state_info("CCO")
        assert "エラー" in info["ionization_note"]

    def test_with_mock_pka_acidic_anion(self):
        """pH > pKa_acid ならアニオン形優位"""
        from backend.chem.protonation import get_protonation_state_info

        mock_model = mock.MagicMock()
        mock_model.get_acidic_macro_pka.return_value = 4.0
        mock_model.get_basic_macro_pka.side_effect = Exception("no basic pKa")

        with mock.patch("backend.chem.protonation._get_unipka_model", return_value=mock_model):
            info = get_protonation_state_info("CC(=O)O", ph=7.4)

        assert info["dominant_form_at_ph"] == "anion"
        assert "アニオン" in info["ionization_note"]
        assert info["pka_acidic"] == pytest.approx(4.0)

    def test_with_mock_pka_basic_cation(self):
        """pH < pKa_base ならカチオン形優位"""
        from backend.chem.protonation import get_protonation_state_info

        mock_model = mock.MagicMock()
        mock_model.get_acidic_macro_pka.side_effect = Exception("no acidic pKa")
        mock_model.get_basic_macro_pka.return_value = 10.0

        with mock.patch("backend.chem.protonation._get_unipka_model", return_value=mock_model):
            info = get_protonation_state_info("CCN", ph=7.4)

        assert info["dominant_form_at_ph"] == "cation"
        assert "カチオン" in info["ionization_note"]

    def test_with_mock_pka_neutral(self):
        """pH < pKa_acid かつ pH > pKa_base なら中性形"""
        from backend.chem.protonation import get_protonation_state_info

        mock_model = mock.MagicMock()
        mock_model.get_acidic_macro_pka.return_value = 10.0
        mock_model.get_basic_macro_pka.return_value = 3.0

        with mock.patch("backend.chem.protonation._get_unipka_model", return_value=mock_model):
            info = get_protonation_state_info("CCO", ph=7.0)

        assert info["dominant_form_at_ph"] == "neutral"

    def test_with_mock_pka_zwitterion(self):
        """pH > pKa_acid かつ pH < pKa_base なら双性イオン"""
        from backend.chem.protonation import get_protonation_state_info

        mock_model = mock.MagicMock()
        mock_model.get_acidic_macro_pka.return_value = 2.0
        mock_model.get_basic_macro_pka.return_value = 9.5

        with mock.patch("backend.chem.protonation._get_unipka_model", return_value=mock_model):
            info = get_protonation_state_info("[NH3+]CC(=O)[O-]", ph=7.0)

        assert info["dominant_form_at_ph"] == "zwitterion"
        assert "双性" in info["ionization_note"]


# ===================================================================
# _get_unipka_model: スレッドロック・リトライ機構
# ===================================================================

class TestGetUnipkaModel:
    def setup_method(self):
        """各テスト前にモジュールキャッシュをリセット"""
        import backend.chem.protonation as pmod
        pmod._unipka_model_cache = None

    def test_cache_reuse(self):
        """2回呼び出しても同一インスタンスが返ること（キャッシュ）"""
        import backend.chem.protonation as pmod

        mock_model = mock.MagicMock()
        call_count = 0

        def mock_unipka(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_model

        with mock.patch("backend.chem.protonation.UnipKa", mock_unipka, create=True):
            with mock.patch.dict("sys.modules", {"unipka": mock.MagicMock(UnipKa=mock_unipka)}):
                pmod._unipka_model_cache = None
                m1 = pmod._get_unipka_model()
                m2 = pmod._get_unipka_model()

        assert m1 is m2

    def test_permission_error_retried(self):
        """WinError 32 の場合にリトライして最終的に成功する"""
        import backend.chem.protonation as pmod

        mock_model = mock.MagicMock()
        attempt = [0]

        class MockUnipKa:
            def __init__(self, *args, **kwargs):
                attempt[0] += 1
                if attempt[0] < 3:
                    raise PermissionError(f"[WinError 32] locked (attempt {attempt[0]})")

        mock_module = mock.MagicMock()
        mock_module.UnipKa = MockUnipKa

        with mock.patch.dict("sys.modules", {"unipka": mock_module}):
            pmod._unipka_model_cache = None
            model = pmod._get_unipka_model(max_retries=5, retry_delay=0.0)

        assert model is not None
        assert attempt[0] == 3

    def test_permission_error_exhausted_raises(self):
        """全リトライ失敗時は PermissionError を発生させる"""
        import backend.chem.protonation as pmod

        class MockUnipKa:
            def __init__(self, *args, **kwargs):
                raise PermissionError("[WinError 32] always locked")

        mock_module = mock.MagicMock()
        mock_module.UnipKa = MockUnipKa

        with mock.patch.dict("sys.modules", {"unipka": mock_module}):
            pmod._unipka_model_cache = None
            with pytest.raises(PermissionError):
                pmod._get_unipka_model(max_retries=3, retry_delay=0.0)

    def test_thread_safety(self):
        """
        複数スレッドから同時に _get_unipka_model を呼んでも
        初期化は1度だけ（スレッドロックの検証）。
        """
        import backend.chem.protonation as pmod

        init_count = [0]
        mock_model = mock.MagicMock()

        class MockUnipKa:
            def __init__(self, *args, **kwargs):
                import time
                time.sleep(0.02)
                init_count[0] += 1

        mock_module = mock.MagicMock()
        mock_module.UnipKa = MockUnipKa

        results = []

        def worker():
            with mock.patch.dict("sys.modules", {"unipka": mock_module}):
                m = pmod._get_unipka_model(max_retries=1, retry_delay=0.0)
                results.append(m)

        pmod._unipka_model_cache = None

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert init_count[0] == 1
        assert all(r is results[0] for r in results)


# ===================================================================
# XTB引数の統合テスト（モックで）
# ===================================================================

class TestXTBArgsIntegration:
    """
    XTBAdapterが correct な --chrg/--uhf を生成するかモックで確認。
    実際のxtbバイナリを呼ばずにコマンドライン引数を検証。
    """

    SMILES_CASES = [
        ("[NH4+]",          1,  1,  0),   # アンモニウムイオン: 電荷+1, 閉殻
        ("CC(=O)[O-]",     -1,  1,  0),   # 酢酸アニオン: 電荷-1, 閉殻
        ("CCO",              0,  1,  0),   # エタノール（中性・閉殻）
    ]

    def test_store_resolves_correct_charge(self):
        """ChargeConfigStore が SMILES から正しい電荷を解決する"""
        store = ChargeConfigStore()
        store.default.auto_charge_from_smiles = True

        for smi, expected_charge, _, _ in self.SMILES_CASES:
            resolved = store.resolve_charge(smi)
            assert resolved == expected_charge, (
                f"SMILES={smi}: expected charge={expected_charge}, got={resolved}"
            )

    def test_xtb_args_from_store(self):
        """ChargeConfigStore の設定から to_xtb_args で正しく変換できる"""
        store = ChargeConfigStore()
        store.default.auto_charge_from_smiles = False
        store.default.formal_charge = -2
        store.default.spin_multiplicity = 3

        cfg = store.get_config("anything")
        args = cfg.to_xtb_args(charge_override=-2)

        assert "--chrg" in args
        assert "-2" in args
        assert "--uhf" in args
        uhf_idx = args.index("--uhf")
        assert args[uhf_idx + 1] == "2"  # spin_multiplicity=3 -> uhf=2
