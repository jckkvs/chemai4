"""
tests/test_charge_extended.py

電荷設定モジュールの拡張テスト（エッジケース網羅）。

カバー対象:
  - _parse_xtb_output(): XTB出力パース（正常/部分出力/空出力/Mulliken電荷）
  - _smiles_to_xyz(): SMILES→XYZ 変換（正常/荷電/不正SMILES）
  - RDKitAdapter: Gasteiger部分電荷テスト（q_max/q_min/q_range等）
  - ChargeConfigStore: エッジケース（二重登録/空文字/大量設定）
  - MoleculeChargeConfig: 追加バリデーション
  - protonation: _max_deprotonate/_max_protonate 深層テスト
"""
from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from backend.chem.charge_config import (
    ChargeConfigStore,
    MoleculeChargeConfig,
    _read_smiles_formal_charge,
)

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

requires_rdkit = pytest.mark.skipif(
    not RDKIT_AVAILABLE, reason="RDKit未インストール"
)


# ═══════════════════════════════════════════════════════════════════
# _parse_xtb_output: XTB出力パーサの単体テスト
# ═══════════════════════════════════════════════════════════════════

# 典型的な GFN2-xTB 出力のスニペット
# パーサの条件:
#   "homo-lumo gap" in line_l                   → xtb_HomoLumoGap
#   "total energy" in line_l and "Eh" in line   → xtb_TotalEnergy
#   "| homo" in line_l and "eV" in line         → xtb_HomoEnergy
#   "| lumo" in line_l and "eV" in line         → xtb_LumoEnergy
#   "| total" in line_l and "debye" in rest     → xtb_DipoleMoment
#   "mulliken" in line_l and "charge" in line_l → Mulliken電荷ブロック開始

_XTB_OUTPUT_NORMAL = """\
   HOMO-LUMO GAP             12.8274 eV
   total energy             -5.070451 Eh
 | HOMO       | -0.51262 |  -13.9492 | eV   |
 | LUMO       | -0.04144 |   -1.1276 | eV   |
 | total |    0.000    0.001    0.002    1.234 Debye
 Mulliken/CM5 charges
     1  C   6  -0.123  -0.089
     2  H   1   0.041   0.022
     3  H   1   0.041   0.022
     4  H   1   0.041   0.022
     5  H   1   0.041   0.022
 total charge = 0.0000
"""

_XTB_OUTPUT_PARTIAL = """\
   total energy             -12.345678 Eh
   HOMO-LUMO gap              5.4321 eV
"""

_XTB_OUTPUT_EMPTY = ""

_XTB_OUTPUT_NO_HOMO_LUMO = """\
   total energy             -5.070451 Eh
"""

_XTB_OUTPUT_MULLIKEN_ONLY = """\
 Mulliken/CM5 charges
     1  N   7  -0.500  -0.400
     2  H   1   0.200   0.150
     3  H   1   0.200   0.150
     4  H   1   0.100   0.100
 total charge = 0.0000
"""


class TestParseXtbOutput:
    """_parse_xtb_output() の単体テスト。"""

    @pytest.fixture(autouse=True)
    def _import_parser(self):
        from backend.chem.xtb_adapter import _parse_xtb_output
        self.parse = _parse_xtb_output

    def test_normal_output_total_energy(self):
        """全電子エネルギーの抽出"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        assert "xtb_TotalEnergy" in r
        assert r["xtb_TotalEnergy"] == pytest.approx(-5.070451, abs=1e-5)

    def test_normal_output_homo_lumo_gap(self):
        """HOMO-LUMOギャップの抽出"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        assert "xtb_HomoLumoGap" in r
        assert r["xtb_HomoLumoGap"] == pytest.approx(12.8274, abs=0.01)

    def test_normal_output_homo_energy(self):
        """HOMOエネルギーの抽出"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        assert "xtb_HomoEnergy" in r
        assert r["xtb_HomoEnergy"] == pytest.approx(-13.9492, abs=0.01)

    def test_normal_output_lumo_energy(self):
        """LUMOエネルギーの抽出"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        assert "xtb_LumoEnergy" in r
        assert r["xtb_LumoEnergy"] == pytest.approx(-1.1276, abs=0.01)

    def test_normal_output_koopmans_derived(self):
        """Koopmans定理由来のIP/EA/求電子性"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        homo = r["xtb_HomoEnergy"]
        lumo = r["xtb_LumoEnergy"]
        assert "xtb_IonizationPotential" in r
        assert r["xtb_IonizationPotential"] == pytest.approx(-homo, abs=0.01)
        assert "xtb_ElectronAffinity" in r
        assert r["xtb_ElectronAffinity"] == pytest.approx(-lumo, abs=0.01)
        # 求電子性: μ²/(2η), μ=(IP+EA)/2, η=(IP-EA)/2
        ip, ea = -homo, -lumo
        mu = (ip + ea) / 2.0
        eta = (ip - ea) / 2.0
        expected_electrophilicity = mu**2 / (2.0 * eta) if eta > 0 else None
        if expected_electrophilicity is not None:
            assert "xtb_Electrophilicity" in r
            assert r["xtb_Electrophilicity"] == pytest.approx(
                expected_electrophilicity, abs=0.01
            )

    def test_normal_output_dipole(self):
        """双極子モーメントの抽出"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        assert "xtb_DipoleMoment" in r
        assert r["xtb_DipoleMoment"] == pytest.approx(1.234, abs=0.01)

    def test_normal_output_mulliken_charges(self):
        """Mulliken電荷統計の抽出（5原子のメタン）"""
        r = self.parse(_XTB_OUTPUT_NORMAL)
        assert "xtb_MullikenChargeMax" in r
        assert "xtb_MullikenChargeMin" in r
        assert "xtb_MullikenChargeMean" in r
        assert "xtb_MullikenChargeStd" in r
        # C=-0.123, H=0.041×4 なので
        assert r["xtb_MullikenChargeMin"] == pytest.approx(-0.123, abs=0.01)
        assert r["xtb_MullikenChargeMax"] == pytest.approx(0.041, abs=0.01)

    def test_partial_output(self):
        """HOMO/LUMO行がない部分出力でもクラッシュしない"""
        r = self.parse(_XTB_OUTPUT_PARTIAL)
        assert "xtb_TotalEnergy" in r
        assert r["xtb_TotalEnergy"] == pytest.approx(-12.345678, abs=1e-4)
        assert "xtb_HomoLumoGap" in r
        # HOMO/LUMO個別が無いのでKoopmans由来は計算不可
        assert "xtb_IonizationPotential" not in r

    def test_empty_output(self):
        """空出力は空辞書を返す"""
        r = self.parse(_XTB_OUTPUT_EMPTY)
        assert r == {}

    def test_no_homo_lumo_lines(self):
        """HOMO-LUMO行が無い出力"""
        r = self.parse(_XTB_OUTPUT_NO_HOMO_LUMO)
        assert "xtb_TotalEnergy" in r
        assert "xtb_HomoEnergy" not in r
        assert "xtb_LumoEnergy" not in r

    def test_mulliken_only_output(self):
        """Mulliken電荷のみの出力"""
        r = self.parse(_XTB_OUTPUT_MULLIKEN_ONLY)
        assert "xtb_MullikenChargeMax" in r
        assert "xtb_MullikenChargeMin" in r
        # N=-0.500, H=0.200, 0.200, 0.100
        assert r["xtb_MullikenChargeMin"] == pytest.approx(-0.500, abs=0.01)
        assert r["xtb_MullikenChargeMax"] == pytest.approx(0.200, abs=0.01)

    def test_garbage_input_no_crash(self):
        """ランダム文字列でクラッシュしない"""
        r = self.parse("This is garbage text\nwith no useful info\n" * 10)
        assert isinstance(r, dict)


# ═══════════════════════════════════════════════════════════════════
# _smiles_to_xyz: SMILES→XYZ 変換テスト
# ═══════════════════════════════════════════════════════════════════

class TestSmilesToXyz:
    """_smiles_to_xyz() の単体テスト。"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.chem.xtb_adapter import _smiles_to_xyz
        self.fn = _smiles_to_xyz

    @requires_rdkit
    def test_methane(self):
        """メタン CH4 → 5原子のXYZ"""
        xyz = self.fn("C")
        assert xyz is not None
        lines = xyz.strip().split("\n")
        n_atoms = int(lines[0].strip())
        assert n_atoms == 5  # C + 4H

    @requires_rdkit
    def test_ethanol(self):
        """エタノール CCO → 9原子のXYZ"""
        xyz = self.fn("CCO")
        assert xyz is not None
        lines = xyz.strip().split("\n")
        n_atoms = int(lines[0].strip())
        assert n_atoms == 9  # 2C + 1O + 6H

    @requires_rdkit
    def test_cation_ammonium(self):
        """荷電分子 [NH4+] → 5原子のXYZ"""
        xyz = self.fn("[NH4+]", charge=1)
        assert xyz is not None
        lines = xyz.strip().split("\n")
        n_atoms = int(lines[0].strip())
        assert n_atoms == 5  # N + 4H

    @requires_rdkit
    def test_anion_acetate(self):
        """アニオン CC(=O)[O-] → XYZ生成"""
        xyz = self.fn("CC(=O)[O-]", charge=-1)
        assert xyz is not None

    def test_invalid_smiles_returns_none(self):
        """不正SMILES → None"""
        xyz = self.fn("INVALID_SMILES###")
        assert xyz is None

    def test_empty_smiles_returns_none(self):
        """空文字列 → None"""
        xyz = self.fn("")
        assert xyz is None

    @requires_rdkit
    def test_xyz_format_has_header(self):
        """XYZフォーマット: 1行目=原子数, 2行目=コメント"""
        xyz = self.fn("C")
        lines = xyz.strip().split("\n")
        assert lines[0].strip().isdigit()
        assert "Generated from SMILES" in lines[1]

    @requires_rdkit
    def test_xyz_coordinates_are_floats(self):
        """各原子行の座標が浮動小数点数であること"""
        xyz = self.fn("C")
        lines = xyz.strip().split("\n")
        n_atoms = int(lines[0])
        for line in lines[2 : 2 + n_atoms]:
            parts = line.split()
            assert len(parts) >= 4
            for coord in parts[1:4]:
                float(coord)  # ValueError が出なければOK

    @requires_rdkit
    def test_benzene_6_carbons(self):
        """ベンゼン c1ccccc1 → 12原子（6C + 6H）"""
        xyz = self.fn("c1ccccc1")
        assert xyz is not None
        lines = xyz.strip().split("\n")
        n_atoms = int(lines[0].strip())
        assert n_atoms == 12


# ═══════════════════════════════════════════════════════════════════
# RDKitAdapter: Gasteiger 部分電荷テスト
# ═══════════════════════════════════════════════════════════════════

class TestRDKitAdapterGasteigerCharges:

    @requires_rdkit
    def test_gasteiger_columns_present(self):
        """compute_gasteiger=True のとき q_max/q_min/q_range/q_std/q_abs_mean が出力される"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        adp = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
        result = adp.compute(["CCO"])
        cols = result.descriptors.columns.tolist()
        for expected in ["gasteiger_q_max", "gasteiger_q_min",
                         "gasteiger_q_range", "gasteiger_q_std", "gasteiger_q_abs_mean"]:
            assert expected in cols, f"{expected} が出力されていません"

    @requires_rdkit
    def test_gasteiger_disabled_no_columns(self):
        """compute_gasteiger=False のとき Gasteiger列が出力されない"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        adp = RDKitAdapter(compute_fp=False, compute_gasteiger=False)
        result = adp.compute(["CCO"])
        cols = result.descriptors.columns.tolist()
        gasteiger_cols = [c for c in cols if "gasteiger" in c.lower()]
        assert len(gasteiger_cols) == 0

    @requires_rdkit
    def test_gasteiger_q_range_nonnegative(self):
        """q_range = q_max - q_min >= 0"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        adp = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
        result = adp.compute(["CCO", "c1ccccc1", "CC(=O)O"])
        df = result.descriptors
        for idx in range(len(df)):
            q_range = df["gasteiger_q_range"].iloc[idx]
            assert q_range >= 0, f"index {idx}: q_range={q_range} は負"

    @requires_rdkit
    def test_gasteiger_acetic_acid_has_large_range(self):
        """酢酸 CC(=O)O: O原子とH原子の電荷差が大きいので q_range > 0.1"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        adp = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
        result = adp.compute(["CC(=O)O"])
        q_range = result.descriptors["gasteiger_q_range"].iloc[0]
        assert q_range > 0.1

    @requires_rdkit
    def test_gasteiger_with_charge_config_store(self):
        """charge_config_store を渡してもクラッシュしない"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        store = ChargeConfigStore()
        adp = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
        result = adp.compute(["CCO", "[NH4+]"], charge_config_store=store)
        assert result.descriptors.shape[0] == 2

    @requires_rdkit
    def test_invalid_smiles_in_batch(self):
        """不正SMILESを含むバッチでもクラッシュしない"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        adp = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
        result = adp.compute(["CCO", "INVALID###", "c1ccccc1"])
        assert result.descriptors.shape[0] == 3
        assert 1 in result.failed_indices

    @requires_rdkit
    def test_physicochemical_descriptors_present(self):
        """主要物理化学記述子（MolWt, LogP, TPSA等）が出力される"""
        from backend.chem.rdkit_adapter import RDKitAdapter
        adp = RDKitAdapter(compute_fp=False)
        result = adp.compute(["CCO"])
        cols = result.descriptors.columns.tolist()
        for expected in ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors", "TPSA"]:
            assert expected in cols


# ═══════════════════════════════════════════════════════════════════
# ChargeConfigStore: 追加エッジケース
# ═══════════════════════════════════════════════════════════════════

class TestChargeConfigStoreEdgeCases:

    def test_overwrite_per_molecule(self):
        """同じSMILESで二度set_per_molecule → 後の設定が有効"""
        store = ChargeConfigStore()
        cfg1 = MoleculeChargeConfig(formal_charge=1)
        cfg2 = MoleculeChargeConfig(formal_charge=-1)
        store.set_per_molecule("CCO", cfg1)
        store.set_per_molecule("CCO", cfg2)
        assert store.get_config("CCO").formal_charge == -1

    def test_empty_string_smiles(self):
        """空文字列SMILESに対してresolve_chargeが例外を投げない"""
        store = ChargeConfigStore()
        store.default.auto_charge_from_smiles = True
        charge = store.resolve_charge("")
        assert charge == 0  # RDKitで解析失敗→デフォルト0

    def test_many_per_molecule_settings(self):
        """100件の個別設定を登録・取得"""
        store = ChargeConfigStore()
        for i in range(100):
            smi = f"C{'C' * i}"
            cfg = MoleculeChargeConfig(formal_charge=i % 5 - 2)
            store.set_per_molecule(smi, cfg)
        assert len(store.per_molecule) == 100
        # "CCCC" = "C" + "C"*3 → i=3 → formal_charge = 3%5-2 = 1
        assert store.get_config("CCCC").formal_charge == 1

    def test_default_is_independent_copy(self):
        """default変更がper_moleculeに影響しない"""
        store = ChargeConfigStore()
        cfg = MoleculeChargeConfig(formal_charge=1)
        store.set_per_molecule("CCO", cfg)
        store.default.formal_charge = -5
        assert store.get_config("CCO").formal_charge == 1

    def test_resolve_spin_unknown_smiles(self):
        """未登録SMILESはデフォルトスピンを返す"""
        store = ChargeConfigStore()
        store.default = MoleculeChargeConfig(spin_multiplicity=3)
        assert store.resolve_spin("UNKNOWN") == 3


# ═══════════════════════════════════════════════════════════════════
# MoleculeChargeConfig: 追加バリデーション・エッジケース
# ═══════════════════════════════════════════════════════════════════

class TestMoleculeChargeConfigEdgeCases:

    def test_high_spin_multiplicity(self):
        """スピン多重度=10（高スピン遷移金属錯体）"""
        cfg = MoleculeChargeConfig(spin_multiplicity=10)
        assert cfg.uhf == 9

    def test_formal_charge_boundary_exact_10(self):
        """ちょうど±10は有効"""
        cfg_p = MoleculeChargeConfig(formal_charge=10)
        cfg_m = MoleculeChargeConfig(formal_charge=-10)
        assert cfg_p.formal_charge == 10
        assert cfg_m.formal_charge == -10

    def test_formal_charge_boundary_11_raises(self):
        """±11は無効"""
        with pytest.raises(ValueError):
            MoleculeChargeConfig(formal_charge=11)
        with pytest.raises(ValueError):
            MoleculeChargeConfig(formal_charge=-11)

    def test_to_xtb_args_high_spin(self):
        """スピン多重度=5のXTB引数"""
        cfg = MoleculeChargeConfig(
            formal_charge=-2,
            spin_multiplicity=5,
            auto_charge_from_smiles=False,
        )
        args = cfg.to_xtb_args()
        assert args == ["--chrg", "-2", "--uhf", "4"]

    def test_charge_override_zero_still_sets_chrg(self):
        """charge_override=0でも--chrg 0は出力される"""
        cfg = MoleculeChargeConfig(formal_charge=3)
        args = cfg.to_xtb_args(charge_override=0)
        assert "--chrg" in args
        idx = args.index("--chrg")
        assert args[idx + 1] == "0"

    def test_default_factory_each_call_independent(self):
        """MoleculeChargeConfig.default() は毎回新しいインスタンス"""
        c1 = MoleculeChargeConfig.default()
        c2 = MoleculeChargeConfig.default()
        assert c1 is not c2
        c1.formal_charge = 5
        assert c2.formal_charge == 0


# ═══════════════════════════════════════════════════════════════════
# _read_smiles_formal_charge: 追加エッジケース
# ═══════════════════════════════════════════════════════════════════

class TestReadSmilesFormalChargeEdgeCases:

    @requires_rdkit
    def test_multi_cation(self):
        """Ca2+ → +2"""
        charge = _read_smiles_formal_charge("[Ca+2]")
        assert charge == 2

    @requires_rdkit
    def test_phosphate_anion(self):
        """リン酸ジアニオン"""
        charge = _read_smiles_formal_charge("[O-]P(=O)([O-])O")
        assert charge == -2

    @requires_rdkit
    def test_neutral_aromatic(self):
        """ピリジン c1ccncc1 → 0"""
        charge = _read_smiles_formal_charge("c1ccncc1")
        assert charge == 0

    def test_numeric_string_returns_0(self):
        """数値文字列は不正SMILES→0"""
        charge = _read_smiles_formal_charge("12345")
        assert charge == 0

    def test_special_chars_returns_0(self):
        """特殊文字は不正SMILES→0"""
        charge = _read_smiles_formal_charge("!@#$%")
        assert charge == 0


# ═══════════════════════════════════════════════════════════════════
# Protonation: _max_deprotonate / _max_protonate 深層テスト
# ═══════════════════════════════════════════════════════════════════

class TestProtonationMaxModes:

    @requires_rdkit
    def test_max_deprotonate_carboxylic_acid(self):
        """酢酸 CC(=O)O の最大脱プロトン化"""
        from backend.chem.protonation import apply_protonation
        cfg = MoleculeChargeConfig(protonate_mode="max_acid")
        result = apply_protonation("CC(=O)O", cfg)
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    @requires_rdkit
    def test_max_protonate_amine(self):
        """メチルアミン CCN の最大プロトン化"""
        from backend.chem.protonation import apply_protonation
        cfg = MoleculeChargeConfig(protonate_mode="max_base")
        result = apply_protonation("CCN", cfg)
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    @requires_rdkit
    def test_max_deprotonate_phenol(self):
        """フェノール c1ccccc1O の最大脱プロトン化"""
        from backend.chem.protonation import apply_protonation
        cfg = MoleculeChargeConfig(protonate_mode="max_acid")
        result = apply_protonation("c1ccccc1O", cfg)
        assert result is not None

    @requires_rdkit
    def test_max_protonate_pyridine(self):
        """ピリジン c1ccncc1 の最大プロトン化"""
        from backend.chem.protonation import apply_protonation
        cfg = MoleculeChargeConfig(protonate_mode="max_base")
        result = apply_protonation("c1ccncc1", cfg)
        assert result is not None

    @requires_rdkit
    def test_neutral_aspirin(self):
        """アスピリンの中性化"""
        from backend.chem.protonation import apply_protonation
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = apply_protonation(aspirin, cfg)
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0

    @requires_rdkit
    def test_neutral_sodium_acetate_salt(self):
        """酢酸ナトリウム [Na+].CC(=O)[O-] → 中性化＋脱塩"""
        from backend.chem.protonation import apply_protonation
        cfg = MoleculeChargeConfig(protonate_mode="neutral")
        result = apply_protonation("[Na+].CC(=O)[O-]", cfg)
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.GetFormalCharge(mol) == 0


# ═══════════════════════════════════════════════════════════════════
# XTBAdapter: compute()のモックテスト
# ═══════════════════════════════════════════════════════════════════

class TestXTBAdapterComputeMock:
    """xtbバイナリ不要 — subprocessをモックしてcompute()のロジックをテスト"""

    @requires_rdkit
    def test_compute_passes_chrg_arg(self):
        """ChargeConfigStoreの電荷がsubprocess.runのコマンドに--chrgとして渡される"""
        from backend.chem.xtb_adapter import XTBAdapter

        store = ChargeConfigStore()
        store.default.auto_charge_from_smiles = False
        store.default.formal_charge = 2
        store.default.spin_multiplicity = 3

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.append(cmd)
            result = mock.MagicMock()
            result.returncode = 0
            result.stdout = _XTB_OUTPUT_NORMAL
            result.stderr = ""
            return result

        adp = XTBAdapter(gfn=2)
        with mock.patch("backend.chem.xtb_adapter.shutil.which", return_value="/usr/bin/xtb"):
            with mock.patch("backend.chem.xtb_adapter.subprocess.run", side_effect=mock_run):
                adp.compute(["C"], charge_config_store=store)

        assert len(captured_cmd) > 0
        cmd = captured_cmd[0]
        assert "--chrg" in cmd
        chrg_idx = cmd.index("--chrg")
        assert cmd[chrg_idx + 1] == "2"
        assert "--uhf" in cmd
        uhf_idx = cmd.index("--uhf")
        assert cmd[uhf_idx + 1] == "2"  # spin=3 → uhf=2

    @requires_rdkit
    def test_compute_timeout_handled(self):
        """subprocessタイムアウト時にクラッシュしない"""
        import subprocess
        from backend.chem.xtb_adapter import XTBAdapter

        adp = XTBAdapter(gfn=2)
        with mock.patch("backend.chem.xtb_adapter.shutil.which", return_value="/usr/bin/xtb"):
            with mock.patch(
                "backend.chem.xtb_adapter.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="xtb", timeout=120),
            ):
                result = adp.compute(["C"])

        assert len(result.failed_indices) == 1
        assert result.descriptors.shape[0] == 1
