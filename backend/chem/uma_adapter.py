# -*- coding: utf-8 -*-
"""
backend/chem/uma_adapter.py

Meta FAIR の UMA (Universal Model for Atoms) による量子化学記述子生成アダプタ。

UMA は OMol25 データセット（1億+量子化学計算）で学習された
DFTレベル精度の分子特性予測モデル。CPU でも高速推論可能。

引用: Meta FAIR, UMA: Universal Model for Atoms (2024)
      https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/

依存関係:
  - fairchem-core >= 2.17.0  (`pip install fairchem-core`)
  - HuggingFace アカウント + アクセストークン
    (https://huggingface.co/facebook/UMA にアクセス申請)
  - RDKit（SMILES → 3D 変換用）

インストール手順:
  1. pip install fairchem-core
  2. huggingface-cli login   # アクセストークンを入力
  3. 初回実行時にモデルが自動ダウンロードされます
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# UMA から取得する記述子の定義
_UMA_DESCRIPTORS = {
    "uma_TotalEnergy":        "分子の全電子エネルギー [eV]",
    "uma_ForceMax":           "原子間力の最大値 [eV/Å]",
    "uma_ForceMean":          "原子間力の平均値 [eV/Å]",
    "uma_ForceStd":           "原子間力の標準偏差 [eV/Å]",
    "uma_StressMax":          "応力テンソルの最大固有値 [eV/Å³]（該当時）",
    "uma_EnergyPerAtom":      "1原子あたりのエネルギー [eV/atom]",
    "uma_Dipole":             "双極子モーメント [e·Å]（該当時）",
}


def _smiles_to_ase_atoms(smiles: str) -> "Any | None":
    """
    SMILES → RDKit → 3D 構造最適化 → ASE Atoms 変換。

    Returns:
        ase.Atoms オブジェクト、変換失敗時は None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from ase import Atoms

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        # 3D 座標生成
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            return None

        # 力場最適化
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass  # 座標はある

        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
        positions = np.array(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z] for i in range(n_atoms)]
        )

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info["charge"] = Chem.GetFormalCharge(mol)
        atoms.info["spin"] = 1  # singlet default

        # 十分な真空を追加（周期境界条件なし）
        atoms.center(vacuum=10.0)
        atoms.pbc = False

        return atoms
    except Exception as e:
        logger.debug(f"SMILES→ASE変換失敗 ({smiles}): {e}")
        return None


class UMAAdapter(BaseChemAdapter):
    """
    Meta UMA (Universal Model for Atoms) による分子記述子生成。

    UMA は DFT レベルの精度で分子のエネルギー・力等を予測する
    ニューラルネットワークモデルです。CPU でも動作し、
    1分子あたり数秒で高精度な量子化学的記述子を提供します。

    Attributes:
        model_name: 使用するモデル名 ("uma-s-1p2" = Small最新版)
        device: 計算デバイス ("cpu" or "cuda")
    """

    def __init__(
        self,
        model_name: str = "uma-s-1p2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._predictor = None
        self._calc = None

    @property
    def name(self) -> str:
        return "uma"

    @property
    def description(self) -> str:
        return (
            "Meta FAIR UMA (Universal Model for Atoms) — "
            "DFTレベル精度の汎用分子特性予測。CPU動作対応。"
            "pip install fairchem-core + HuggingFace認証が必要。"
        )

    def is_available(self) -> bool:
        """fairchem-core と RDKit がインストール済みかチェック。"""
        try:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator  # noqa: F401
            from rdkit import Chem  # noqa: F401
            from ase import Atoms  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model_loaded(self) -> None:
        """モデルを遅延ロードする。"""
        if self._predictor is None:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator

            logger.info(
                f"UMA モデル '{self.model_name}' をロード中 (device={self.device})..."
            )
            self._predictor = pretrained_mlip.get_predict_unit(
                self.model_name, device=self.device
            )
            self._calc = FAIRChemCalculator(
                self._predictor, task_name="omol"
            )
            logger.info("UMA モデルのロードが完了しました。")

    def compute(
        self,
        smiles_list: list[str],
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILESリストから UMA 記述子を計算する。

        各分子について:
        1. SMILES → RDKit → 3D座標 → ASE Atoms
        2. UMA モデルでエネルギー・力等を予測
        3. 統計量を記述子として抽出
        """
        self._require_available()
        logger.info(f"🪐 UMA (Universal Model for Atoms) 計算を開始します（{len(smiles_list)}分子）...")
        self._ensure_model_loaded()

        rows: list[dict[str, float]] = []
        failed_indices: list[int] = []

        for i, smi in enumerate(smiles_list):
            try:
                atoms = _smiles_to_ase_atoms(smi)
                if atoms is None:
                    failed_indices.append(i)
                    rows.append({k: np.nan for k in _UMA_DESCRIPTORS})
                    continue

                # UMA Calculator を設定して計算実行
                atoms.calc = self._calc

                energy = atoms.get_potential_energy()  # eV
                forces = atoms.get_forces()  # (N, 3) eV/Å
                n_atoms = len(atoms)

                force_norms = np.linalg.norm(forces, axis=1)

                row = {
                    "uma_TotalEnergy": float(energy),
                    "uma_ForceMax": float(np.max(force_norms)),
                    "uma_ForceMean": float(np.mean(force_norms)),
                    "uma_ForceStd": float(np.std(force_norms)),
                    "uma_EnergyPerAtom": float(energy / n_atoms) if n_atoms > 0 else np.nan,
                }

                # 応力テンソル（PBC の場合のみ有効、分子では通常 N/A）
                try:
                    stress = atoms.get_stress()
                    row["uma_StressMax"] = float(np.max(np.abs(stress))) if stress is not None else np.nan
                except Exception:
                    row["uma_StressMax"] = np.nan

                # 双極子モーメント（モデルが対応していれば）
                try:
                    dipole = atoms.get_dipole_moment()
                    row["uma_Dipole"] = float(np.linalg.norm(dipole)) if dipole is not None else np.nan
                except Exception:
                    row["uma_Dipole"] = np.nan

                rows.append(row)
                if (i + 1) % 10 == 0 or i == len(smiles_list) - 1:
                    logger.info(f"UMA: {i + 1}/{len(smiles_list)} 分子計算完了")

            except Exception as e:
                logger.warning(f"UMA 計算失敗 (idx={i}, smi={smi}): {e}")
                failed_indices.append(i)
                rows.append({k: np.nan for k in _UMA_DESCRIPTORS})

        df = pd.DataFrame(rows)
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={"model_name": self.model_name, "device": self.device},
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        """UMA記述子のメタデータを返す。"""
        return [
            DescriptorMetadata(
                name=name,
                meaning=meaning,
                is_count=False,
                is_binary=False,
                description=f"UMA ({self.model_name}) による量子化学的記述子",
            )
            for name, meaning in _UMA_DESCRIPTORS.items()
        ]

    def get_descriptor_names(self) -> list[str]:
        """計算可能な記述子名のリストを返す。"""
        return list(_UMA_DESCRIPTORS.keys())
