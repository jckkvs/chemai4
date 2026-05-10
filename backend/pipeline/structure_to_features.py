"""
backend/pipeline/structure_to_features.py

SMILES → RDKit 3D構造生成 → xTB最適化 → ML派生特徴量抽出
の統合パイプライン。

既存の xtb_adapter.py, rdkit_adapter.py, charge_config.py を
**そのまま**使用し、xtb_ml_features.py による派生特徴量を追加する。
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_structure_feature_pipeline(
    smiles_list: list[str],
    charge_config_store: Any | None = None,
    xtb_kwargs: dict[str, Any] | None = None,
    feature_config: Any | None = None,
    include_rdkit_2d: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """
    SMILES → 3D構造 → xTB最適化 → ML特徴量抽出 のフルパイプラインを実行する。

    既存のアダプター (XTBAdapter, RDKitAdapter) はそのまま使用。
    派生特徴量は XTBMLFeatureExtractor で追加計算する。

    Args:
        smiles_list: 入力SMILESのリスト。
        charge_config_store: ChargeConfigStore インスタンス（省略可）。
        xtb_kwargs: XTBAdapter.compute() に渡す追加引数
            (calc_type, convergence, solvent 等)。
        feature_config: XTBFeatureConfig インスタンス（省略可）。
        include_rdkit_2d: True のとき、RDKit 2D記述子も結合する。
        progress_callback: 進捗コールバック
            ``(current_index, total, message_str)``。

    Returns:
        行=分子, 列=特徴量 の DataFrame。

    Raises:
        RuntimeError: xTBバイナリが見つからない場合。

    Example::

        from backend.pipeline.structure_to_features import run_structure_feature_pipeline
        df = run_structure_feature_pipeline(["CCO", "c1ccccc1"])
    """
    from backend.chem.xtb_adapter import XTBAdapter
    from backend.chem.xtb_ml_features import XTBMLFeatureExtractor

    # ── 初期化 ──
    xtb = XTBAdapter(**(xtb_kwargs or {}))

    if not xtb.is_available():
        raise RuntimeError(
            "xTBバイナリが見つかりません。"
            "conda install -c conda-forge xtb でインストールしてください。"
        )

    extractor = XTBMLFeatureExtractor(feature_config)

    if progress_callback:
        progress_callback(0, len(smiles_list), "xTB 計算を開始...")

    # ── Step 1: xTB 計算（既存の adapter をそのまま使用）──
    xtb_result = xtb.compute(
        smiles_list,
        charge_config_store=charge_config_store,
        **(xtb_kwargs or {}),
    )

    if progress_callback:
        progress_callback(
            len(smiles_list), len(smiles_list),
            f"xTB 計算完了（成功率: {xtb_result.success_rate:.0%}）",
        )

    # ── Step 2: 最適化後座標の取得（metadata から）──
    optimized_coords = xtb_result.metadata.get("optimized_coords", [])
    xyz_list: list[np.ndarray | None] = []
    atoms_list: list[list[int] | None] = []

    for coord_info in optimized_coords:
        if coord_info is not None and "coords" in coord_info:
            xyz_list.append(coord_info["coords"])
            atoms_list.append(coord_info.get("atomic_numbers"))
        else:
            xyz_list.append(None)
            atoms_list.append(None)

    # optimized_coords が不足している場合の補完
    while len(xyz_list) < len(smiles_list):
        xyz_list.append(None)
        atoms_list.append(None)

    # ── Step 3: 派生ML特徴量の抽出 ──
    xtb_records = xtb_result.descriptors.to_dict("records")
    ml_features_df = extractor.batch_extract(xtb_records, xyz_list, atoms_list)

    # ── Step 4: 既存xTB記述子 + 派生特徴量の結合 ──
    combined = pd.concat(
        [xtb_result.descriptors.reset_index(drop=True),
         ml_features_df.reset_index(drop=True)],
        axis=1,
    )

    # ── Step 5: RDKit 2D記述子の結合（オプション）──
    if include_rdkit_2d:
        try:
            from backend.chem.rdkit_adapter import RDKitAdapter

            rdkit = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
            if rdkit.is_available():
                rdkit_result = rdkit.compute(
                    smiles_list,
                    charge_config_store=charge_config_store,
                )
                combined = pd.concat(
                    [rdkit_result.descriptors.reset_index(drop=True),
                     combined.reset_index(drop=True)],
                    axis=1,
                )
                logger.info(
                    "RDKit 2D記述子 %d列を結合しました",
                    rdkit_result.descriptors.shape[1],
                )
        except Exception as e:
            logger.warning("RDKit 2D記述子の追加に失敗: %s", e)

    logger.info(
        "構造→特徴量パイプライン完了: %d分子 × %d特徴量",
        combined.shape[0], combined.shape[1],
    )
    return combined
