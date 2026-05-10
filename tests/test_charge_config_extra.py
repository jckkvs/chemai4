"""
tests/test_charge_config_extra.py

charge_config.py のカバレッジ改善テスト。
MoleculeChargeConfig, ChargeConfigStore, _read_smiles_formal_charge を網羅。
"""
from __future__ import annotations

import pytest

from backend.chem.charge_config import (
    MoleculeChargeConfig,
    ChargeConfigStore,
    _read_smiles_formal_charge,
)


# ============================================================
# MoleculeChargeConfig
# ============================================================

class TestMoleculeChargeConfig:
    def test_defaults(self):
        cfg = MoleculeChargeConfig()
        assert cfg.formal_charge == 0
        assert cfg.spin_multiplicity == 1
        assert cfg.protonate_mode == "as_is"
        assert cfg.partial_charge_model == "gasteiger"
        assert cfg.auto_charge_from_smiles is True

    def test_uhf(self):
        cfg = MoleculeChargeConfig(spin_multiplicity=3)
        assert cfg.uhf == 2

    def test_to_xtb_args_default(self):
        cfg = MoleculeChargeConfig()
        args = cfg.to_xtb_args()
        assert args == ["--chrg", "0"]

    def test_to_xtb_args_with_spin(self):
        cfg = MoleculeChargeConfig(formal_charge=1, spin_multiplicity=2)
        args = cfg.to_xtb_args()
        assert "--chrg" in args
        assert "1" in args
        assert "--uhf" in args
        assert "1" in args

    def test_to_xtb_args_charge_override(self):
        cfg = MoleculeChargeConfig(formal_charge=0)
        args = cfg.to_xtb_args(charge_override=-1)
        assert args == ["--chrg", "-1"]

    def test_invalid_spin(self):
        with pytest.raises(ValueError, match="spin_multiplicity"):
            MoleculeChargeConfig(spin_multiplicity=0)

    def test_invalid_charge(self):
        with pytest.raises(ValueError, match="formal_charge"):
            MoleculeChargeConfig(formal_charge=100)

    def test_default_factory(self):
        cfg = MoleculeChargeConfig.default()
        assert cfg.formal_charge == 0
        assert cfg.spin_multiplicity == 1

    def test_for_radical(self):
        cfg = MoleculeChargeConfig.for_radical(charge=-1)
        assert cfg.spin_multiplicity == 2
        assert cfg.formal_charge == -1

    def test_at_physiological_ph(self):
        cfg = MoleculeChargeConfig.at_physiological_ph()
        assert cfg.protonate_mode == "auto_ph"
        assert cfg.ph == 7.4


# ============================================================
# ChargeConfigStore
# ============================================================

class TestChargeConfigStore:
    def test_defaults(self):
        store = ChargeConfigStore()
        assert store.default.formal_charge == 0
        assert len(store.per_molecule) == 0

    def test_get_config_default(self):
        store = ChargeConfigStore()
        cfg = store.get_config("CCO")
        assert cfg.formal_charge == 0

    def test_set_per_molecule(self):
        store = ChargeConfigStore()
        custom = MoleculeChargeConfig(formal_charge=1)
        store.set_per_molecule("CCO", custom)
        cfg = store.get_config("CCO")
        assert cfg.formal_charge == 1

    def test_get_config_fallback(self):
        store = ChargeConfigStore()
        store.set_per_molecule("CCO", MoleculeChargeConfig(formal_charge=1))
        cfg = store.get_config("CC")  # Not per_molecule, should be default
        assert cfg.formal_charge == 0

    def test_resolve_spin(self):
        store = ChargeConfigStore()
        assert store.resolve_spin("CCO") == 1

    def test_resolve_charge_auto(self):
        """auto_charge_from_smiles=True 時の resolve_charge"""
        store = ChargeConfigStore()
        # RDKitがインストールされていない場合は0が返る
        charge = store.resolve_charge("CCO")
        assert isinstance(charge, int)

    def test_resolve_charge_manual(self):
        store = ChargeConfigStore(
            default=MoleculeChargeConfig(
                formal_charge=2,
                auto_charge_from_smiles=False,
            )
        )
        charge = store.resolve_charge("CCO")
        assert charge == 2


# ============================================================
# _read_smiles_formal_charge
# ============================================================

class TestReadSmilesFormalCharge:
    def test_neutral(self):
        charge = _read_smiles_formal_charge("CCO")
        assert charge == 0

    def test_invalid_smiles(self):
        charge = _read_smiles_formal_charge("INVALID_SMILES_XYZ")
        assert charge == 0  # RDKit fails → returns 0
