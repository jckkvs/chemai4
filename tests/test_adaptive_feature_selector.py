"""
tests/test_adaptive_feature_selector.py

AdaptiveFeatureSelector のユニットテスト。
"""
import pytest

from backend.chem.adaptive_feature_selector import (
    AdaptiveFeatureSelector,
    FeatureSelectionResult,
)


@pytest.fixture
def selector() -> AdaptiveFeatureSelector:
    return AdaptiveFeatureSelector()


# ── 基本テスト ──

def test_select_general_default(selector):
    result = selector.select()
    assert isinstance(result, FeatureSelectionResult)
    assert len(result.selected_features) > 0
    assert result.task_type == "general"


def test_select_solubility(selector):
    result = selector.select(task_type="solubility", n_molecules=50)
    assert "rdkit_2d" in result.selected_features
    assert "morgan_fp" in result.selected_features
    assert "vibrational" not in result.selected_features  # excluded
    assert "conformer_ensemble" not in result.selected_features  # excluded


def test_select_reactivity(selector):
    result = selector.select(task_type="reactivity", n_molecules=20)
    assert "xtb_opt" in result.selected_features
    assert "xtb_ml_derived" in result.selected_features
    assert "fukui_approx" in result.selected_features
    assert result.requires_xtb is True
    assert result.requires_opt is True


def test_unknown_task_falls_back_to_general(selector):
    result = selector.select(task_type="unknown_task_xyz")
    assert result.task_type == "unknown_task_xyz"
    assert len(result.selected_features) > 0


# ── 予算制約 ──

def test_tight_budget_excludes_expensive(selector):
    result = selector.select(
        task_type="general",
        max_time_per_mol_s=0.05,  # 非常にタイト
    )
    # RDKit系（高速）のみ含まれるはず
    assert "rdkit_2d" in result.selected_features
    assert "xtb_opt" not in result.selected_features
    assert result.budget_met is True


def test_generous_budget_includes_more(selector):
    result = selector.select(
        task_type="general",
        max_time_per_mol_s=1000,
    )
    assert len(result.selected_features) >= 4  # 多くの特徴量が選択される


# ── xTB利用不可 ──

def test_no_xtb_excludes_quantum(selector):
    result = selector.select(
        task_type="reactivity",
        xtb_available=False,
    )
    for f_name in result.selected_features:
        assert "xtb" not in f_name.lower() or "rdkit" in f_name.lower()
    assert result.requires_xtb is False


# ── 強制include/exclude ──

def test_force_include(selector):
    result = selector.select(
        task_type="general",
        force_include=["fukui_approx"],
    )
    assert "fukui_approx" in result.selected_features


def test_force_exclude(selector):
    result = selector.select(
        task_type="solubility",
        force_exclude=["rdkit_2d"],
    )
    assert "rdkit_2d" not in result.selected_features


# ── 結果の数値プロパティ ──

def test_estimated_time_positive(selector):
    result = selector.select(n_molecules=100)
    assert result.estimated_time_per_mol_s >= 0
    assert result.estimated_total_minutes >= 0


def test_budget_met_flag(selector):
    result = selector.select(max_time_per_mol_s=0.001)
    # 超タイト予算ではmust_haveがあるため超過可能
    assert isinstance(result.budget_met, bool)


# ── メタデータ ──

def test_available_tasks(selector):
    tasks = selector.available_tasks
    assert "solubility" in tasks
    assert "reactivity" in tasks
    assert "general" in tasks


def test_available_features(selector):
    features = selector.available_features
    assert "rdkit_2d" in features
    assert "xtb_opt" in features


def test_get_task_description(selector):
    desc = selector.get_task_description("solubility")
    assert desc is not None
    assert "溶解度" in desc


def test_get_cost_summary(selector):
    summary = selector.get_cost_summary()
    assert isinstance(summary, list)
    assert len(summary) > 0
    assert "name" in summary[0]
    assert "time_per_mol_s" in summary[0]


# ── 結果の整合性 ──

def test_no_duplicates_in_selection(selector):
    result = selector.select(
        task_type="general",
        force_include=["rdkit_2d"],  # already in must_have
    )
    assert len(result.selected_features) == len(set(result.selected_features))
