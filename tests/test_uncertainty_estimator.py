"""
tests/test_uncertainty_estimator.py

UncertaintyEstimator のユニットテスト。
"""
import math

import pytest

from backend.chem.uncertainty_estimator import ConfidenceReport, UncertaintyEstimator


@pytest.fixture
def estimator() -> UncertaintyEstimator:
    return UncertaintyEstimator()


@pytest.fixture
def good_xtb_result() -> dict:
    """収束性良好な典型的有機分子のxTB結果。"""
    return {
        "xtb_TotalEnergy": -10.123,
        "xtb_HomoEnergy": -6.5,
        "xtb_LumoEnergy": -1.2,
        "xtb_HomoLumoGap": 5.3,
        "xtb_DipoleMoment": 2.45,
        "xtb_MullikenChargeMax": 0.35,
        "xtb_MullikenChargeMin": -0.42,
        "xtb_MullikenChargeMean": 0.0,
        "xtb_MullikenChargeStd": 0.15,
    }


@pytest.fixture
def unstable_xtb_result() -> dict:
    """HOMO-LUMOギャップが小さい不安定系。"""
    return {
        "xtb_TotalEnergy": -5.0,
        "xtb_HomoEnergy": -3.0,
        "xtb_LumoEnergy": -2.7,
        "xtb_HomoLumoGap": 0.3,
        "xtb_MullikenChargeMean": 0.0,
    }


# ── 基本テスト ──

def test_good_result_high_confidence(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result)
    assert report.overall_confidence > 0.8
    assert len(report.warnings) == 0


def test_empty_result_zero_confidence(estimator):
    report = estimator.evaluate({})
    assert report.overall_confidence < 0.1
    assert report.convergence_score == 0.0


def test_unstable_result_low_electronic(estimator, unstable_xtb_result):
    report = estimator.evaluate(unstable_xtb_result)
    assert report.electronic_stability < 0.5
    assert any("HOMO-LUMO" in w for w in report.warnings)


# ── 収束性 ──

def test_convergence_fallback(estimator, good_xtb_result):
    report = estimator.evaluate(
        good_xtb_result, calc_type_used="sp", convergence_fallback=True
    )
    assert report.convergence_score < 1.0
    assert report.convergence_score > 0.3


def test_convergence_missing_essentials(estimator):
    partial = {"xtb_TotalEnergy": -10.0}
    report = estimator.evaluate(partial)
    assert report.convergence_score < 1.0


# ── 電子状態安定性 ──

def test_large_gap_full_stability(estimator):
    result = {"xtb_HomoLumoGap": 5.0}
    report = estimator.evaluate(result)
    assert report.electronic_stability == 1.0


def test_negative_gap_very_low(estimator):
    result = {"xtb_HomoLumoGap": -0.5}
    report = estimator.evaluate(result)
    assert report.electronic_stability <= 0.1


def test_missing_gap_neutral(estimator):
    report = estimator.evaluate({"xtb_TotalEnergy": -10.0})
    assert report.electronic_stability == 0.5


# ── 電荷一貫性 ──

def test_charge_consistent(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result, formal_charge=0)
    assert report.charge_consistency >= 0.9


# ── 記述子完全性 ──

def test_complete_descriptors(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result)
    assert report.descriptor_completeness == 1.0


def test_nan_descriptors(estimator):
    result = {
        "xtb_TotalEnergy": float("nan"),
        "xtb_HomoEnergy": -6.5,
        "xtb_LumoEnergy": None,
    }
    report = estimator.evaluate(result)
    assert report.descriptor_completeness < 1.0


# ── 立体配座代表性 ──

def test_rigid_molecule_high_repr(estimator, good_xtb_result):
    energies = [-10.123, -10.1225, -10.1228]
    report = estimator.evaluate(good_xtb_result, conformer_energies=energies)
    assert report.conformer_representativeness > 0.8


def test_flexible_molecule_low_repr(estimator, good_xtb_result):
    energies = [-10.123, -10.05, -9.98, -9.90]
    report = estimator.evaluate(good_xtb_result, conformer_energies=energies)
    assert report.conformer_representativeness < 0.8


def test_single_conformer_full_repr(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result, conformer_energies=[-10.123])
    assert report.conformer_representativeness == 1.0


# ── to_features ──

def test_to_features(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result)
    features = report.to_features()
    assert "conf_overall" in features
    assert "conf_convergence" in features
    assert all(0.0 <= v <= 1.0 for v in features.values())


def test_to_features_custom_prefix(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result)
    features = report.to_features(prefix="trust_")
    assert "trust_overall" in features
    assert "conf_overall" not in features


# ── バッチ処理 ──

def test_batch_evaluate(estimator, good_xtb_result, unstable_xtb_result):
    reports = estimator.batch_evaluate(
        [good_xtb_result, unstable_xtb_result, {}]
    )
    assert len(reports) == 3
    assert reports[0].overall_confidence > reports[1].overall_confidence
    assert reports[2].overall_confidence < 0.1


# ── 総合スコアの数値特性 ──

def test_overall_is_bounded(estimator, good_xtb_result):
    report = estimator.evaluate(good_xtb_result)
    assert 0.0 <= report.overall_confidence <= 1.0
