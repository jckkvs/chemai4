"""
backend/chem/mixture_ratio_converter.py

混合比率の内部変換モジュール。
重量比 ↔ モル比 の相互変換 + 「それ以外の比率」の扱い。

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RatioConversionResult:
    """変換結果。"""
    weight_fractions: list[float]   # 重量分率 (合計≈1.0)
    mole_fractions: list[float]     # モル分率 (合計≈1.0)
    molecular_weights: list[float]  # 各成分の分子量
    conversion_warnings: list[str] = field(default_factory=list)


class MixtureRatioConverter:
    """
    混合比率の相互変換と検証。

    使い方::

        converter = MixtureRatioConverter(
            smiles_list=["CCO", "CCCO"],
            ratio_values=[70.0, 30.0],
            input_ratio_type="weight",
        )
        result = converter.convert()
        print(result.weight_fractions)  # [0.7, 0.3]
        print(result.mole_fractions)    # [0.753, 0.247] (分子量で変換)
    """

    def __init__(
        self,
        smiles_list: list[str],
        ratio_values: list[float],
        input_ratio_type: Literal["weight", "mole", "other"],
        molecular_weights: list[float] | None = None,
    ):
        """
        Args:
            smiles_list: 各成分のSMILES。
            ratio_values: 入力された比率値（正規化前）。
            input_ratio_type: 入力比率の種類。
            molecular_weights: 分子量リスト（省略時はRDKitで自動計算）。
        """
        if len(smiles_list) != len(ratio_values):
            raise ValueError(
                f"smiles_list({len(smiles_list)})とratio_values"
                f"({len(ratio_values)})の長さが一致しません"
            )

        self.smiles_list = smiles_list
        self.ratio_values = np.array(ratio_values, dtype=float)
        self.input_ratio_type = input_ratio_type

        # 分子量の取得
        if molecular_weights is not None:
            self.molecular_weights = np.array(molecular_weights, dtype=float)
        else:
            self.molecular_weights = np.array(
                [self._get_mol_weight(s) for s in smiles_list],
                dtype=float,
            )

    def convert(self) -> RatioConversionResult:
        """
        入力比率から重量分率・モル分率の両方を計算する。

        Returns:
            RatioConversionResult
        """
        warnings: list[str] = []

        # 比率の正規化（合計=1.0に）
        total = np.sum(self.ratio_values)
        if total <= 1e-10:
            raise ValueError("比率の合計がゼロまたは負です")
        raw_ratios = self.ratio_values / total

        if self.input_ratio_type == "weight":
            weight_fractions = raw_ratios.copy()
            # 重量比 → モル比: n_i = w_i / M_i
            mole_nums = raw_ratios / self.molecular_weights
            mole_fractions = mole_nums / np.sum(mole_nums)

        elif self.input_ratio_type == "mole":
            mole_fractions = raw_ratios.copy()
            # モル比 → 重量比: w_i = x_i * M_i
            weight_nums = raw_ratios * self.molecular_weights
            weight_fractions = weight_nums / np.sum(weight_nums)

        elif self.input_ratio_type == "other":
            # 「それ以外の比率」: 自動変換不可
            weight_fractions = raw_ratios.copy()
            mole_fractions = raw_ratios.copy()
            warnings.append(
                "'other'比率は重量比/モル比に自動変換できません。"
                "特徴量計算時は手動で加重方法を選択してください。"
            )
        else:
            raise ValueError(f"未知のratio_type: {self.input_ratio_type}")

        # 数値安定性のための微小値クリッピング + 再正規化
        weight_fractions = np.clip(weight_fractions, 1e-10, 1.0)
        mole_fractions = np.clip(mole_fractions, 1e-10, 1.0)
        weight_fractions /= np.sum(weight_fractions)
        mole_fractions /= np.sum(mole_fractions)

        return RatioConversionResult(
            weight_fractions=weight_fractions.tolist(),
            mole_fractions=mole_fractions.tolist(),
            molecular_weights=self.molecular_weights.tolist(),
            conversion_warnings=warnings,
        )

    @staticmethod
    def validate_ratio_input(
        ratio_values: list[float],
        min_components: int = 2,
    ) -> list[str]:
        """比率入力のバリデーション。"""
        errors: list[str] = []
        if len(ratio_values) < min_components:
            errors.append(f"成分数は{min_components}以上が必要です")
        if any(r <= 0 for r in ratio_values):
            errors.append("比率値は正の数である必要があります")
        if sum(ratio_values) <= 1e-10:
            errors.append("比率の合計が小さすぎます")
        return errors

    @staticmethod
    def _get_mol_weight(smiles: str) -> float:
        """SMILESから分子量を計算する。"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"不正なSMILES: {smiles}")
            return Descriptors.MolWt(mol)
        except ImportError:
            raise RuntimeError("RDKitが必要です")
