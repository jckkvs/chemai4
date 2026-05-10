"""
tests/test_xtb_ml_features.py

XTBMLFeatureExtractor のユニットテスト。

xTB出力辞書のモックデータを使用して、バックエンドのxtb_adapterに依存せず
派生特徴量の計算ロジックを検証する。
"""
import numpy as np
import pytest

from backend.chem.xtb_ml_features import XTBFeatureConfig, XTBMLFeatureExtractor


# ── テスト用モックデータ ──

@pytest.fixture
def mock_xtb_result() -> dict:
    """xTB計算結果のモック（xtb_adapter._parse_xtb_output の出力形式）。"""
    return {
        "xtb_TotalEnergy": -10.123456,
        "xtb_HomoEnergy": -6.5,          # eV
        "xtb_LumoEnergy": -1.2,          # eV
        "xtb_HomoLumoGap": 5.3,          # eV
        "xtb_DipoleMoment": 2.45,        # Debye
        "xtb_Polarizability": 12.3,      # Bohr³
        "xtb_IonizationPotential": 6.5,  # eV
        "xtb_ElectronAffinity": 1.2,     # eV
        "xtb_Electrophilicity": 2.78,    # eV
        "xtb_MullikenChargeMax": 0.35,
        "xtb_MullikenChargeMin": -0.42,
        "xtb_MullikenChargeMean": 0.0,
        "xtb_MullikenChargeStd": 0.15,
    }


@pytest.fixture
def mock_xyz_coords() -> np.ndarray:
    """3原子分子のモック座標（水分子のような配置）。"""
    return np.array([
        [0.0, 0.0, 0.0],    # O
        [0.96, 0.0, 0.0],   # H
        [-0.24, 0.93, 0.0], # H
    ])


@pytest.fixture
def mock_atomic_numbers() -> list[int]:
    """水分子: O(8), H(1), H(1)"""
    return [8, 1, 1]


# ============================================================
# 基本テスト：デフォルト設定での抽出
# ============================================================

def test_extract_default_config(mock_xtb_result):
    """デフォルト設定で特徴量が抽出されることを確認。"""
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert isinstance(features, dict)
    assert len(features) > 0
    # エネルギー関連
    assert "xtb_ml_TotalEnergy" in features
    assert "xtb_ml_HomoEnergy" in features
    assert "xtb_ml_LumoEnergy" in features
    assert "xtb_ml_Gap" in features
    # 双極子
    assert "xtb_ml_Dipole" in features
    # 概念DFT
    assert "xtb_ml_Hardness" in features
    assert "xtb_ml_Softness" in features
    assert "xtb_ml_Electrophilicity" in features


def test_extract_values_correct(mock_xtb_result):
    """派生特徴量の値が化学的に正しいことを検証。"""
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    # IP = -E_HOMO = 6.5 eV
    # EA = -E_LUMO = 1.2 eV
    # η = (IP - EA) / 2 = (6.5 - 1.2) / 2 = 2.65
    assert abs(features["xtb_ml_Hardness"] - 2.65) < 1e-8

    # S = 1 / (2η) = 1 / 5.3
    assert abs(features["xtb_ml_Softness"] - 1.0 / 5.3) < 1e-8

    # μ = -(IP + EA) / 2 = -(6.5 + 1.2) / 2 = -3.85
    # ω = μ² / (2η) = 3.85² / (2 * 2.65)
    expected_omega = (3.85 ** 2) / (2.0 * 2.65)
    assert abs(features["xtb_ml_Electrophilicity"] - expected_omega) < 1e-8


# ============================================================
# 設定による出力制御
# ============================================================

def test_disable_energy(mock_xtb_result):
    """include_energy=False のときエネルギー系が含まれないことを確認。"""
    config = XTBFeatureConfig(include_energy=False)
    extractor = XTBMLFeatureExtractor(config)
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert "xtb_ml_TotalEnergy" not in features
    assert "xtb_ml_HomoEnergy" not in features
    assert "xtb_ml_Gap" not in features
    # 概念DFTは include_hardness_softness で制御（エネルギーとは独立）
    assert "xtb_ml_Hardness" in features


def test_disable_hardness(mock_xtb_result):
    """include_hardness_softness=False のとき概念DFT指標が含まれないことを確認。"""
    config = XTBFeatureConfig(include_hardness_softness=False, include_electrophilicity=False)
    extractor = XTBMLFeatureExtractor(config)
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert "xtb_ml_Hardness" not in features
    assert "xtb_ml_Softness" not in features
    assert "xtb_ml_Electrophilicity" not in features


def test_include_polarizability(mock_xtb_result):
    """include_polarizability=True のとき分極率が含まれることを確認。"""
    config = XTBFeatureConfig(include_polarizability=True)
    extractor = XTBMLFeatureExtractor(config)
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert "xtb_ml_Polarizability" in features
    assert features["xtb_ml_Polarizability"] == 12.3


def test_custom_prefix(mock_xtb_result):
    """カスタムプレフィックスが反映されることを確認。"""
    config = XTBFeatureConfig(feature_prefix="qchem_")
    extractor = XTBMLFeatureExtractor(config)
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert any(k.startswith("qchem_") for k in features.keys())
    assert not any(k.startswith("xtb_ml_") for k in features.keys())


# ============================================================
# 電荷統計
# ============================================================

def test_charge_statistics(mock_xtb_result):
    """charge_statistics 設定が機能することを確認。"""
    config = XTBFeatureConfig(charge_statistics=["max", "min", "mean", "std"])
    extractor = XTBMLFeatureExtractor(config)
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert features["xtb_ml_ChargeMax"] == 0.35
    assert features["xtb_ml_ChargeMin"] == -0.42
    assert features["xtb_ml_ChargeMean"] == 0.0
    assert features["xtb_ml_ChargeStd"] == 0.15


# ============================================================
# 福井関数近似
# ============================================================

def test_fukui_approx(mock_xtb_result):
    """簡易福井関数近似が計算されることを確認。"""
    config = XTBFeatureConfig(include_fukui_approx=True)
    extractor = XTBMLFeatureExtractor(config)
    features = extractor.extract_from_xtb_output(mock_xtb_result)

    assert "xtb_ml_FukuiElectrophilic" in features
    assert "xtb_ml_FukuiNucleophilic" in features
    assert features["xtb_ml_FukuiElectrophilic"] == abs(-0.42)
    assert features["xtb_ml_FukuiNucleophilic"] == abs(0.35)


# ============================================================
# 3D幾何学的特徴量
# ============================================================

def test_3d_geometry_features(mock_xtb_result, mock_xyz_coords, mock_atomic_numbers):
    """3D座標から幾何特徴量が計算されることを確認。"""
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(
        mock_xtb_result, mock_xyz_coords, mock_atomic_numbers
    )

    assert "xtb_ml_3D_InertiaMin" in features
    assert "xtb_ml_3D_InertiaMid" in features
    assert "xtb_ml_3D_InertiaMax" in features
    assert "xtb_ml_3D_Asphericity" in features
    assert "xtb_ml_3D_MaxDistance" in features
    assert "xtb_ml_3D_MeanDistance" in features
    assert "xtb_ml_3D_MinDistance" in features

    # 慣性の固有値は非負
    assert features["xtb_ml_3D_InertiaMin"] >= 0
    assert features["xtb_ml_3D_InertiaMid"] >= features["xtb_ml_3D_InertiaMin"]
    assert features["xtb_ml_3D_InertiaMax"] >= features["xtb_ml_3D_InertiaMid"]

    # 距離は正
    assert features["xtb_ml_3D_MaxDistance"] > 0
    assert features["xtb_ml_3D_MeanDistance"] > 0
    assert features["xtb_ml_3D_MinDistance"] > 0
    assert features["xtb_ml_3D_MaxDistance"] >= features["xtb_ml_3D_MeanDistance"]


def test_3d_without_coords(mock_xtb_result):
    """座標が提供されない場合、3D特徴量はスキップされることを確認。"""
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(mock_xtb_result, xyz_coordinates=None)

    assert not any("3D_" in k for k in features.keys())


def test_3d_element_statistics(mock_xtb_result, mock_xyz_coords, mock_atomic_numbers):
    """元素別統計（重心距離）が計算されることを確認。"""
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(
        mock_xtb_result, mock_xyz_coords, mock_atomic_numbers
    )

    # H(1)が2原子あるので元素別統計があるはず
    assert "xtb_ml_3D_Elem1_RadiusMean" in features
    # O(8)は1原子しかないので元素別統計は作られない（2原子未満）
    assert "xtb_ml_3D_Elem8_RadiusMean" not in features


# ============================================================
# バッチ処理
# ============================================================

def test_batch_extract(mock_xtb_result, mock_xyz_coords, mock_atomic_numbers):
    """batch_extractでDataFrameが正しい形状で返ることを確認。"""
    extractor = XTBMLFeatureExtractor()

    results_list = [mock_xtb_result, mock_xtb_result, mock_xtb_result]
    xyz_list = [mock_xyz_coords, mock_xyz_coords, mock_xyz_coords]
    atoms_list = [mock_atomic_numbers, mock_atomic_numbers, mock_atomic_numbers]

    df = extractor.batch_extract(results_list, xyz_list, atoms_list)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3  # 3分子
    assert df.shape[1] > 0   # 特徴量が存在
    # 全行で同じ値（同じ入力データなので）
    assert df.iloc[0].equals(df.iloc[1])


def test_batch_extract_no_coords(mock_xtb_result):
    """座標なしでもbatch_extractが機能することを確認。"""
    extractor = XTBMLFeatureExtractor()
    results_list = [mock_xtb_result, mock_xtb_result]

    df = extractor.batch_extract(results_list)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2


# ============================================================
# エッジケース
# ============================================================

def test_empty_xtb_result():
    """空の辞書でもエラーにならないことを確認。"""
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output({})

    assert isinstance(features, dict)
    # エネルギーがないので概念DFTも計算されない
    assert "xtb_ml_Hardness" not in features


def test_partial_xtb_result():
    """HOMOはあるがLUMOがない場合、概念DFTがスキップされることを確認。"""
    partial = {"xtb_HomoEnergy": -6.5}
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(partial)

    assert "xtb_ml_HomoEnergy" in features
    assert "xtb_ml_LumoEnergy" not in features
    assert "xtb_ml_Hardness" not in features  # LUMO不足で計算不能


def test_nan_values():
    """NaN値を含むxtb_resultでもクラッシュしないことを確認。"""
    import math
    nan_result = {
        "xtb_HomoEnergy": float("nan"),
        "xtb_LumoEnergy": -1.2,
        "xtb_DipoleMoment": float("nan"),
    }
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(nan_result)
    # NaN値は特徴量として含まれるが、例外は発生しない
    assert isinstance(features, dict)


def test_single_atom_coords(mock_xtb_result):
    """1原子分子の座標では3D特徴量がスキップされることを確認。"""
    single_atom = np.array([[0.0, 0.0, 0.0]])
    extractor = XTBMLFeatureExtractor()
    features = extractor.extract_from_xtb_output(
        mock_xtb_result, single_atom, [6]
    )
    # 1原子では慣性モーメント・距離計算が無意味なのでスキップ
    assert "xtb_ml_3D_MaxDistance" not in features


# ============================================================
# XTBFeatureConfig のデフォルト値テスト
# ============================================================

def test_config_defaults():
    """XTBFeatureConfig のデフォルト値が正しいことを確認。"""
    config = XTBFeatureConfig()

    assert config.include_energy is True
    assert config.include_orbitals is True
    assert config.include_dipole is True
    assert config.include_polarizability is False
    assert config.charge_statistics is None
    assert config.include_hardness_softness is True
    assert config.include_electrophilicity is True
    assert config.include_fukui_approx is False
    assert config.include_3d_geom is True
    assert config.feature_prefix == "xtb_ml_"


# pandas import for batch tests
import pandas as pd
