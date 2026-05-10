# -*- coding: utf-8 -*-
"""
backend/chem/skfp_adapter.py

scikit-fingerprints アダプタ。
30種以上の分子フィンガープリント（ECFP, MACCS, TopologicalTorsion, Avalon等）を
sklearn互換インターフェースで計算する。

参考: scikit-fingerprints (https://github.com/scikit-fingerprints/scikit-fingerprints)
  pip install scikit-fingerprints
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# 利用可能なフィンガープリントとそのデフォルト設定
_FP_CONFIGS: dict[str, dict[str, Any]] = {
    "ECFP": {"fp_size": 2048, "radius": 2},
    "MACCS": {},
    "TopologicalTorsion": {"fp_size": 2048},
    "Atom Pair": {"fp_size": 2048},
    "Avalon": {"fp_size": 512},
    "RDKit": {"fp_size": 2048},
    "FCFP": {"fp_size": 2048, "radius": 2, "use_features": True},
    "MAP": {"fp_size": 2048},
    "ERG": {},
    "Layered": {"fp_size": 2048},
    "Pattern": {"fp_size": 2048},
    "LINGO": {"fp_size": 1024},
    "Klekota-Roth": {},
    "PhysiochemicalProperties": {},
    "GETAWAY": {},
    "MORSE": {},
    "WHIM": {},
    "Autocorrelation": {},
}


class SkfpAdapter(BaseChemAdapter):
    """
    scikit-fingerprints アダプタ。

    複数のフィンガープリントを同時計算し、結合した DataFrame を返す。
    デフォルトでは ECFP (r=2, 2048bit) + MACCS (167bit) を計算。
    """

    def __init__(
        self,
        fp_types: list[str] | None = None,
        fp_configs: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Args:
            fp_types: 計算するFP名のリスト（Noneで ["ECFP", "MACCS"]）
            fp_configs: FP名 → パラメータ辞書の上書き
        """
        self._fp_types = fp_types or ["ECFP", "MACCS"]
        self._fp_configs = fp_configs or {}

    @property
    def name(self) -> str:
        return "skfp"

    @property
    def description(self) -> str:
        return (
            "scikit-fingerprints: 30種以上の分子フィンガープリント "
            "(ECFP, MACCS, TopologicalTorsion, Avalon等)。"
            "pip install scikit-fingerprints"
        )

    def is_available(self) -> bool:
        try:
            import skfp  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        logger.info(f"🔑 Scikit-FP フィンガープリント計算を開始します（{len(smiles_list)}分子）...")
        import skfp.fingerprints as fps
        from rdkit import Chem

        n = len(smiles_list)
        all_frames: list[pd.DataFrame] = []
        failed_indices: set[int] = set()

        # SMILES → Mol 変換
        mols = []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smi}")
                mols.append(mol)
            except Exception:
                mols.append(None)
                failed_indices.add(i)

        for fp_name in self._fp_types:
            config = {**_FP_CONFIGS.get(fp_name, {}), **self._fp_configs.get(fp_name, {})}

            try:
                # FPクラスを動的に取得
                fp_cls_name = fp_name.replace(" ", "") + "Fingerprint"
                if not hasattr(fps, fp_cls_name):
                    # 名前のバリエーションを試す
                    for attr in dir(fps):
                        if attr.lower() == fp_cls_name.lower():
                            fp_cls_name = attr
                            break

                if hasattr(fps, fp_cls_name):
                    fp_calculator = getattr(fps, fp_cls_name)(**config)
                    # 有効なmolのみ計算
                    valid_mols = [m for m in mols if m is not None]
                    if valid_mols:
                        fp_array = fp_calculator.transform(valid_mols)
                        n_bits = fp_array.shape[1]
                        col_names = [f"{fp_name}_{j}" for j in range(n_bits)]

                        # 全サンプルの結果を構築（失敗はNaN）
                        full_array = np.full((n, n_bits), np.nan)
                        valid_idx = 0
                        for i in range(n):
                            if mols[i] is not None:
                                full_array[i] = fp_array[valid_idx]
                                valid_idx += 1

                        all_frames.append(pd.DataFrame(full_array, columns=col_names))
                    else:
                        logger.warning(f"skfp: {fp_name} — 有効な分子がありません")
                else:
                    logger.warning(f"skfp: {fp_cls_name} クラスが見つかりません")
            except Exception as e:
                logger.warning(f"skfp: {fp_name} 計算エラー: {e}")

        if all_frames:
            descriptors = pd.concat(all_frames, axis=1)
        else:
            descriptors = pd.DataFrame(index=range(n))

        return DescriptorResult(
            descriptors=descriptors,
            smiles_list=smiles_list,
            failed_indices=sorted(failed_indices),
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        _FP_JP = {
            "ECFP": "ECFP (Extended-Connectivity FP)。原子の化学環境の円形探索で分子部分構造をエンコード。類似性検索・QSAR の標準",
            "MACCS": "MACCS構造キー。166種の部分構造パターンの有無。医薬品化学で広く利用",
            "AtomPair": "AtomPair FP。原子対間の距離と原子タイプをエンコード。構造類似性の検出",
            "TopologicalTorsion": "トポロジカルトーション FP。4原子経路の結合パターン。コンフォメーション情報を捕捉",
            "Morgan": "Morgan FP (円形FP)。ECFPの一般化。半径パラメータで部分構造の粒度を制御",
            "RDKitFP": "RDKit FP。分子グラフの経路ベース フィンガープリント",
            "Avalon": "Avalon FP。分子の部分構造パターンの高速エンコーディング",
            "FCFP": "FCFP (Feature-Connectivity FP)。ECFPの拡張。原子の薬理学的特徴 (水素結合等) をエンコード",
            "LayeredFP": "Layered FP。分子グラフの階層的パス情報",
            "PharmacophoreECFP": "PharmacophoreECFP。薬理学的特徴 (HBA, HBD等) をECFP的にエンコード",
        }
        meta = []
        for fp_name in self._fp_types:
            config = {**_FP_CONFIGS.get(fp_name, {}), **self._fp_configs.get(fp_name, {})}
            n_bits = config.get("fp_size", 167 if fp_name == "MACCS" else 2048)
            fp_desc = _FP_JP.get(fp_name, f"{fp_name} フィンガープリント")
            for j in range(n_bits):
                meta.append(DescriptorMetadata(
                    name=f"{fp_name}_{j}",
                    meaning=f"{fp_desc} ビット{j}",
                    is_count=False,
                ))
        return meta
