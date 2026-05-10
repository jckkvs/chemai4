"""
tests/test_conformer_ensemble.py

ConformerEnsembleExtractor のユニットテスト。
"""
import numpy as np
import pytest

from backend.chem.conformer_ensemble_features import (
    ConformerEnsembleConfig,
    ConformerEnsembleExtractor,
)


@pytest.fixture
def extractor() -> ConformerEnsembleExtractor:
    return ConformerEnsembleExtractor()


@pytest.fixture
def three_conformers() -> list[dict]:
    """3つのconformerのxTB結果モック。"""
    return [
        {
            "xtb_TotalEnergy": -10.123,
            "xtb_HomoEnergy": -6.5,
            "xtb_LumoEnergy": -1.2,
            "xtb_HomoLumoGap": 5.3,
            "xtb_DipoleMoment": 2.45,
            "xtb_Polarizability": 12.3,
            "xtb_MullikenChargeMax": 0.35,
            "xtb_MullikenChargeMin": -0.42,
            "xtb_MullikenChargeStd": 0.15,
        },
        {
            "xtb_TotalEnergy": -10.120,
            "xtb_HomoEnergy": -6.4,
            "xtb_LumoEnergy": -1.3,
            "xtb_HomoLumoGap": 5.1,
            "xtb_DipoleMoment": 2.80,
            "xtb_Polarizability": 12.5,
            "xtb_MullikenChargeMax": 0.33,
            "xtb_MullikenChargeMin": -0.40,
            "xtb_MullikenChargeStd": 0.14,
        },
        {
            "xtb_TotalEnergy": -10.118,
            "xtb_HomoEnergy": -6.6,
            "xtb_LumoEnergy": -1.1,
            "xtb_HomoLumoGap": 5.5,
            "xtb_DipoleMoment": 2.10,
            "xtb_Polarizability": 12.1,
            "xtb_MullikenChargeMax": 0.37,
            "xtb_MullikenChargeMin": -0.44,
            "xtb_MullikenChargeStd": 0.16,
        },
    ]


# ── 基本テスト ──

def test_extract_returns_dict(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert isinstance(features, dict)
    assert len(features) > 0


def test_n_conformers_recorded(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert features["ens_n_conformers"] == 3.0


def test_single_conformer_returns_empty(extractor):
    features = extractor.extract([{"xtb_TotalEnergy": -10.0}])
    assert features == {}


def test_empty_list_returns_empty(extractor):
    features = extractor.extract([])
    assert features == {}


# ── エネルギー統計 ──

def test_energy_stats_present(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert "ens_energy_mean" in features
    assert "ens_energy_std" in features
    assert "ens_energy_range" in features
    assert "ens_energy_min" in features
    assert "ens_energy_range_kcal" in features


def test_energy_mean_correct(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    expected_mean = np.mean([-10.123, -10.120, -10.118])
    assert abs(features["ens_energy_mean"] - expected_mean) < 1e-8


def test_energy_range_kcal_correct(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    range_hartree = 10.123 - 10.118
    range_kcal = range_hartree * 627.509
    assert abs(features["ens_energy_range_kcal"] - range_kcal) < 0.01


def test_boltzmann_weighted_mean(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert "ens_energy_boltz_mean" in features
    # 最低エネルギーconformerが最も重いので、boltz_mean < arithmetic mean
    assert features["ens_energy_boltz_mean"] <= features["ens_energy_mean"]


# ── 軌道揺らぎ ──

def test_orbital_fluctuation_present(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert "ens_homo_mean" in features
    assert "ens_homo_std" in features
    assert "ens_lumo_mean" in features
    assert "ens_gap_mean" in features


def test_homo_std_correct(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    expected_std = np.std([-6.5, -6.4, -6.6])
    assert abs(features["ens_homo_std"] - expected_std) < 1e-8


# ── 電荷揺らぎ ──

def test_charge_fluctuation_present(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert "ens_qmax_mean" in features
    assert "ens_qmax_std" in features
    assert "ens_qmin_mean" in features


# ── 形状安定性 ──

def test_shape_stability_present(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert "ens_dipole_mean" in features
    assert "ens_dipole_std" in features
    assert "ens_dipole_cv" in features


def test_cv_is_positive(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert features["ens_dipole_cv"] > 0


# ── Boltzmannエントロピー ──

def test_boltzmann_entropy(extractor, three_conformers):
    features = extractor.extract(three_conformers)
    assert "ens_boltzmann_entropy" in features
    assert features["ens_boltzmann_entropy"] > 0


# ── 設定による制御 ──

def test_disable_energy_stats(three_conformers):
    config = ConformerEnsembleConfig(include_energy_stats=False)
    extractor = ConformerEnsembleExtractor(config)
    features = extractor.extract(three_conformers)
    assert "ens_energy_mean" not in features
    assert "ens_homo_mean" in features  # orbital is still enabled


def test_disable_orbital(three_conformers):
    config = ConformerEnsembleConfig(include_orbital_fluctuation=False)
    extractor = ConformerEnsembleExtractor(config)
    features = extractor.extract(three_conformers)
    assert "ens_homo_mean" not in features


def test_disable_all_but_energy(three_conformers):
    config = ConformerEnsembleConfig(
        include_orbital_fluctuation=False,
        include_charge_fluctuation=False,
        include_shape_stability=False,
    )
    extractor = ConformerEnsembleExtractor(config)
    features = extractor.extract(three_conformers)
    assert "ens_energy_mean" in features
    assert "ens_homo_mean" not in features
    assert "ens_qmax_mean" not in features
    assert "ens_dipole_mean" not in features


def test_custom_prefix(three_conformers):
    config = ConformerEnsembleConfig(feature_prefix="conf_")
    extractor = ConformerEnsembleExtractor(config)
    features = extractor.extract(three_conformers)
    assert any(k.startswith("conf_") for k in features)
    assert not any(k.startswith("ens_") for k in features)


# ── エッジケース ──

def test_missing_energy_in_one_conformer(extractor):
    results = [
        {"xtb_TotalEnergy": -10.0, "xtb_HomoEnergy": -6.0},
        {"xtb_HomoEnergy": -6.1},  # energy missing
    ]
    features = extractor.extract(results)
    # Energy stats should still work with available data
    assert isinstance(features, dict)


def test_all_same_energy(extractor):
    results = [
        {"xtb_TotalEnergy": -10.0},
        {"xtb_TotalEnergy": -10.0},
        {"xtb_TotalEnergy": -10.0},
    ]
    features = extractor.extract(results)
    assert features["ens_energy_std"] == 0.0
    assert features["ens_energy_range"] == 0.0
