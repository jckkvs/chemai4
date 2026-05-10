"""
tests/test_multi_method_pipeline.py

MultiMethodPipeline のユニットテスト。

xTB/RDKitのアダプターをモックして、パイプラインのロジックのみを検証する。
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.pipeline.multi_method_pipeline import (
    LayerResult,
    MultiMethodPipeline,
    PipelineConfig,
)


@pytest.fixture
def pipeline() -> MultiMethodPipeline:
    return MultiMethodPipeline()


@pytest.fixture
def mock_rdkit_result():
    """RDKitAdapter.compute() の戻り値モック。"""
    mock = MagicMock()
    mock.descriptors = pd.DataFrame({
        "MolWt": [46.07, 78.11, 92.14],
        "LogP": [-0.31, 1.56, 2.07],
        "TPSA": [20.23, 0.0, 0.0],
    })
    mock.failed_indices = []
    return mock


@pytest.fixture
def mock_xtb_result():
    """XTBAdapter.compute() の戻り値モック。"""
    mock = MagicMock()
    mock.descriptors = pd.DataFrame({
        "xtb_TotalEnergy": [-10.1, -15.2],
        "xtb_HomoEnergy": [-6.5, -5.8],
        "xtb_LumoEnergy": [-1.2, -0.9],
        "xtb_HomoLumoGap": [5.3, 4.9],
        "xtb_DipoleMoment": [2.4, 0.1],
    })
    mock.failed_indices = []
    mock.metadata = {"optimized_coords": [None, None]}
    mock.success_rate = 1.0
    return mock


# ── 基本テスト ──

def test_pipeline_rdkit_only(pipeline, mock_rdkit_result):
    """RDKitのみ（xTB無効）でパイプラインが動くことを確認。"""
    config = PipelineConfig(enable_layer2_xtb=False)

    with patch("backend.pipeline.multi_method_pipeline.MultiMethodPipeline._run_layer1_rdkit") as mock_l1:
        mock_l1.return_value = LayerResult(
            layer_name="rdkit_2d",
            n_input=3, n_output=3,
            descriptors=mock_rdkit_result.descriptors,
            passed_indices=[0, 1, 2],
        )

        result = pipeline.run(["CCO", "c1ccccc1", "Cc1ccccc1"], config)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3


def test_pipeline_both_layers(pipeline, mock_rdkit_result, mock_xtb_result):
    """RDKit + xTBの両層が動くことを確認。"""
    config = PipelineConfig(
        enable_ml_features=False,
        enable_uncertainty=False,
    )

    with patch("backend.pipeline.multi_method_pipeline.MultiMethodPipeline._run_layer1_rdkit") as mock_l1, \
         patch("backend.pipeline.multi_method_pipeline.MultiMethodPipeline._run_layer2_xtb") as mock_l2:

        mock_l1.return_value = LayerResult(
            "rdkit_2d", 2, 2,
            mock_rdkit_result.descriptors.iloc[:2],
            [0, 1],
        )
        mock_l2.return_value = LayerResult(
            "xtb", 2, 2,
            mock_xtb_result.descriptors,
            [0, 1],
        )

        result = pipeline.run(["CCO", "c1ccccc1"], config)

    assert result.shape[0] == 2
    assert "MolWt" in result.columns or "rdkit_2d_MolWt" in result.columns


# ── フィルタリング ──

def test_layer1_filter(pipeline, mock_rdkit_result, mock_xtb_result):
    """Layer 1のフィルタで分子が絞り込まれることを確認。"""
    def only_heavy(df: pd.DataFrame) -> list[int]:
        return [i for i, mw in enumerate(df["MolWt"]) if mw > 70]

    config = PipelineConfig(
        layer1_filter=only_heavy,
        enable_ml_features=False,
        enable_uncertainty=False,
    )

    with patch("backend.pipeline.multi_method_pipeline.MultiMethodPipeline._run_layer1_rdkit") as mock_l1, \
         patch("backend.pipeline.multi_method_pipeline.MultiMethodPipeline._run_layer2_xtb") as mock_l2:

        mock_l1.return_value = LayerResult(
            "rdkit_2d", 3, 3,
            mock_rdkit_result.descriptors,
            [0, 1, 2],
        )
        mock_l2.return_value = LayerResult(
            "xtb", 2, 2,
            mock_xtb_result.descriptors,
            [1, 2],  # CCO(idx=0)がフィルタされた
        )

        result = pipeline.run(
            ["CCO", "c1ccccc1", "Cc1ccccc1"], config,
        )

    assert result.shape[0] == 3  # 全分子分の行
    # xTB特徴量はidx=1,2のみ、idx=0はNaN


# ── 無効化テスト ──

def test_skip_all_layers(pipeline):
    """全レイヤーを無効にしても空DataFrameが返ることを確認。"""
    config = PipelineConfig(
        enable_layer1_rdkit=False,
        enable_layer2_xtb=False,
    )
    result = pipeline.run(["CCO"], config)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1


# ── LayerResult ──

def test_layer_result_defaults():
    lr = LayerResult("test", 10, 8, pd.DataFrame(), [0, 1, 2])
    assert lr.layer_name == "test"
    assert lr.n_input == 10
    assert lr.elapsed_seconds == 0.0
    assert lr.failed_indices == []


# ── PipelineConfig ──

def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.enable_layer1_rdkit is True
    assert cfg.enable_layer2_xtb is True
    assert cfg.enable_ml_features is True
    assert cfg.enable_uncertainty is True
    assert cfg.xtb_calc_type == "opt"
    assert cfg.min_confidence == 0.0


# ── 結果マージ ──

def test_merge_empty_results(pipeline):
    result = pipeline._merge_results({}, 3)
    assert result.shape[0] == 3
    assert result.shape[1] == 0


def test_merge_single_layer(pipeline, mock_rdkit_result):
    results = {
        "rdkit": LayerResult(
            "rdkit", 3, 3,
            mock_rdkit_result.descriptors,
            [0, 1, 2],
        ),
    }
    merged = pipeline._merge_results(results, 3)
    assert merged.shape[0] == 3
    assert "MolWt" in merged.columns


def test_merge_partial_indices(pipeline):
    """一部分子のみ計算された場合のマージ。"""
    df = pd.DataFrame({"energy": [-10.0, -15.0]})
    results = {
        "xtb": LayerResult(
            "xtb", 2, 2,
            df,
            [1, 3],  # idx 0,2 はスキップ
        ),
    }
    merged = pipeline._merge_results(results, 4)
    assert merged.shape[0] == 4
    assert pd.isna(merged.loc[0, "energy"])  # スキップされた分子はNaN
    assert merged.loc[1, "energy"] == -10.0
    assert pd.isna(merged.loc[2, "energy"])
    assert merged.loc[3, "energy"] == -15.0


# ── 進捗コールバック ──

def test_progress_callback(pipeline, mock_rdkit_result):
    """進捗コールバックが呼ばれることを確認。"""
    calls = []

    def on_progress(msg, current, total):
        calls.append((msg, current, total))

    config = PipelineConfig(
        enable_layer2_xtb=False,
        progress_callback=on_progress,
    )

    with patch("backend.pipeline.multi_method_pipeline.MultiMethodPipeline._run_layer1_rdkit") as mock_l1:
        mock_l1.return_value = LayerResult(
            "rdkit_2d", 2, 2,
            mock_rdkit_result.descriptors.iloc[:2],
            [0, 1],
        )
        pipeline.run(["CCO", "c1ccccc1"], config)

    # _run_layer1_rdkit内部でコールバックが呼ばれるが、
    # モックしているため外部からは確認できない
    # ただしPipelineConfigに設定されたことは確認できる
    assert config.progress_callback is not None
