"""
backend/chem/mixture_feature_extractor.py

混合物系の特徴量抽出エンジン。
各成分の単一分子記述子を計算し、物理化学的根拠に基づく
加重平均で混合物特徴量を合成する。

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from backend.chem.descriptor_weighting_classifier import classify_descriptor
from backend.chem.mixture_ratio_converter import MixtureRatioConverter, RatioConversionResult

logger = logging.getLogger(__name__)


@dataclass
class MixtureFeatureResult:
    """混合物特徴量抽出の結果。"""
    mixture_features: dict[str, float]
    component_features: list[dict[str, float]] | None
    conversion_info: dict[str, Any]
    weighting_log: dict[str, str]  # {feature_name: used_weighting_type}
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self, strip_prefix: bool = False) -> pd.DataFrame:
        """混合物特徴量を1行DataFrameとして返す。

        Args:
            strip_prefix: True のとき列名から 'mix_' プレフィックスを除去する。
                           単一化合物データとの混在時に使用。
        """
        df = pd.DataFrame([self.mixture_features])
        if strip_prefix:
            df.columns = [
                c[len('mix_'):] if c.startswith('mix_') else c
                for c in df.columns
            ]
        return df


class MixtureFeatureExtractor:
    """
    混合物の特徴量を成分から合成するエンジン。

    各成分の記述子を個別計算し、物理化学的根拠に基づく
    加重方法（重量比 or モル比）で加重平均を取る。

    使い方::

        extractor = MixtureFeatureExtractor()
        result = extractor.extract(
            components=[
                {"smiles": "CCO", "ratio_value": 70.0, "ratio_unit": "weight"},
                {"smiles": "CCCO", "ratio_value": 30.0, "ratio_unit": "weight"},
            ],
        )
        print(result.mixture_features)  # {"mix_MolWt": 50.28, ...}
    """

    def __init__(
        self,
        user_overrides: dict[str, Literal["weight", "mole"]] | None = None,
    ):
        """
        Args:
            user_overrides: ユーザーによる加重方法の上書き。
                {descriptor_name: "weight" or "mole"}
        """
        self._user_overrides = user_overrides or {}

    def extract(
        self,
        components: list[dict[str, Any]],
        compute_individual: bool = True,
    ) -> MixtureFeatureResult:
        """
        混合物の特徴量を抽出する。

        Args:
            components: 各成分の情報リスト。
                各要素: {"smiles": str, "ratio_value": float,
                          "ratio_unit": "weight"|"mole"|"other",
                          "compound_name": str (optional)}
            compute_individual: True のとき成分ごとの特徴量も返す。

        Returns:
            MixtureFeatureResult
        """
        # 1. 入力バリデーション
        smiles_list = [c["smiles"] for c in components]
        ratio_values = [float(c["ratio_value"]) for c in components]
        ratio_unit = components[0].get("ratio_unit", "weight")

        errors = MixtureRatioConverter.validate_ratio_input(ratio_values)
        if errors:
            raise ValueError(f"比率入力エラー: {'; '.join(errors)}")

        # 2. 比率変換
        converter = MixtureRatioConverter(
            smiles_list, ratio_values, ratio_unit,
        )
        conversion = converter.convert()

        # 3. 各成分の記述子を計算
        component_features_list: list[dict[str, float]] = []
        try:
            from backend.chem.rdkit_adapter import RDKitAdapter

            adapter = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
            if adapter.is_available():
                result = adapter.compute(smiles_list)
                for i in range(len(smiles_list)):
                    if i < len(result.descriptors):
                        row = result.descriptors.iloc[i].to_dict()
                        component_features_list.append(row)
                    else:
                        component_features_list.append({})
            else:
                logger.warning("RDKitが利用不可")
        except Exception as e:
            logger.error("成分記述子の計算に失敗: %s", e)

        # 4. 加重平均で混合物特徴量を合成
        mixture_features: dict[str, float] = {}
        weighting_log: dict[str, str] = {}
        warnings: list[str] = list(conversion.conversion_warnings)

        if component_features_list:
            all_feature_names = set()
            for cf in component_features_list:
                all_feature_names.update(cf.keys())

            for feat_name in sorted(all_feature_names):
                comp_values = []
                for cf in component_features_list:
                    v = cf.get(feat_name)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        comp_values.append(float(v))
                    else:
                        comp_values.append(0.0)

                # 加重方法の決定
                fractions, used_type = self._get_fractions(
                    feat_name, conversion,
                )
                weighting_log[feat_name] = used_type

                # 加重平均
                weighted_val = sum(
                    v * f for v, f in zip(comp_values, fractions)
                )
                mixture_features[f"mix_{feat_name}"] = weighted_val

        return MixtureFeatureResult(
            mixture_features=mixture_features,
            component_features=component_features_list if compute_individual else None,
            conversion_info={
                "weight_fractions": conversion.weight_fractions,
                "mole_fractions": conversion.mole_fractions,
                "molecular_weights": conversion.molecular_weights,
                "input_ratio_type": ratio_unit,
            },
            weighting_log=weighting_log,
            warnings=warnings,
        )

    def extract_batch(
        self,
        mixture_list: list[list[dict[str, Any]]],
    ) -> pd.DataFrame:
        """
        複数の混合物をバッチ処理する。

        Args:
            mixture_list: 混合物リスト。各要素は extract() の components と同形式。

        Returns:
            行=混合物, 列=特徴量 の DataFrame。
        """
        results = []
        for i, components in enumerate(mixture_list):
            try:
                result = self.extract(components, compute_individual=False)
                results.append(result.mixture_features)
            except Exception as e:
                logger.warning("混合物 %d の特徴量抽出に失敗: %s", i, e)
                results.append({})

        return pd.DataFrame(results)

    def _get_fractions(
        self,
        feat_name: str,
        conversion: RatioConversionResult,
    ) -> tuple[list[float], str]:
        """
        特徴量名に対して使用する分率と分類名を返す。

        優先順位: ユーザー上書き > 明示的マッピング > 正規表現 > デフォルト
        """
        # ユーザー上書き
        if feat_name in self._user_overrides:
            override = self._user_overrides[feat_name]
            if override == "weight":
                return conversion.weight_fractions, "weight(user)"
            else:
                return conversion.mole_fractions, "mole(user)"

        # 自動分類
        wtype, _rationale = classify_descriptor(feat_name)
        if wtype == "weight":
            return conversion.weight_fractions, "weight"
        elif wtype == "mole":
            return conversion.mole_fractions, "mole"
        else:
            # context → デフォルトで重量比（安全側）
            return conversion.weight_fractions, "context→weight"
