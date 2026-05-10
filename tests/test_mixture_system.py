"""
tests/test_mixture_system.py

混合物系モジュールのユニットテスト。
- MixtureRatioConverter
- descriptor_weighting_classifier
- MixtureFeatureExtractor
"""
import math

import numpy as np
import pytest

from backend.chem.descriptor_weighting_classifier import (
    EXPLICIT_DESCRIPTOR_MAP,
    classify_all,
    classify_descriptor,
    get_weighting_summary,
)
from backend.chem.mixture_ratio_converter import (
    MixtureRatioConverter,
    RatioConversionResult,
)


# ============================================================
# MixtureRatioConverter テスト
# ============================================================

class TestMixtureRatioConverter:

    def test_weight_to_mole(self):
        """重量比 → モル比の変換。"""
        # エタノール(46.07) : プロパノール(60.10) = 70:30
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "CCCO"],
            ratio_values=[70.0, 30.0],
            input_ratio_type="weight",
        )
        result = conv.convert()

        assert abs(sum(result.weight_fractions) - 1.0) < 1e-8
        assert abs(sum(result.mole_fractions) - 1.0) < 1e-8
        assert abs(result.weight_fractions[0] - 0.7) < 0.01
        # エタノールは軽いのでモル分率はweight分率より大きい
        assert result.mole_fractions[0] > result.weight_fractions[0]

    def test_mole_to_weight(self):
        """モル比 → 重量比の変換。"""
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "CCCO"],
            ratio_values=[1.0, 1.0],
            input_ratio_type="mole",
        )
        result = conv.convert()

        assert abs(result.mole_fractions[0] - 0.5) < 0.01
        assert abs(result.mole_fractions[1] - 0.5) < 0.01
        # エタノール(MW=46)とプロパノール(MW=60) → 重量比はプロパノールが多い
        assert result.weight_fractions[1] > result.weight_fractions[0]

    def test_other_ratio_type_warning(self):
        """'other'タイプは警告が出る。"""
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "O"],
            ratio_values=[2.0, 8.0],
            input_ratio_type="other",
        )
        result = conv.convert()
        assert len(result.conversion_warnings) > 0

    def test_custom_molecular_weights(self):
        """分子量を手動指定できる。"""
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "CCCO"],
            ratio_values=[50, 50],
            input_ratio_type="weight",
            molecular_weights=[46.07, 60.10],
        )
        result = conv.convert()
        assert abs(result.molecular_weights[0] - 46.07) < 0.01

    def test_normalization(self):
        """比率が自動正規化される。"""
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "O"],
            ratio_values=[200, 800],
            input_ratio_type="weight",
        )
        result = conv.convert()
        assert abs(result.weight_fractions[0] - 0.2) < 0.01

    def test_length_mismatch_error(self):
        """SMILES数と比率数の不一致でエラー。"""
        with pytest.raises(ValueError, match="長さが一致しません"):
            MixtureRatioConverter(
                smiles_list=["CCO", "O"],
                ratio_values=[1.0],
                input_ratio_type="weight",
            )

    def test_validate_ratio_input(self):
        """バリデーション関数。"""
        assert len(MixtureRatioConverter.validate_ratio_input([1.0, 2.0])) == 0
        assert len(MixtureRatioConverter.validate_ratio_input([1.0])) > 0  # 成分不足
        assert len(MixtureRatioConverter.validate_ratio_input([1.0, -1.0])) > 0  # 負の値
        assert len(MixtureRatioConverter.validate_ratio_input([0.0, 0.0])) > 0  # 合計ゼロ

    def test_three_components(self):
        """3成分混合。"""
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "CCCO", "O"],
            ratio_values=[50, 30, 20],
            input_ratio_type="weight",
        )
        result = conv.convert()
        assert len(result.weight_fractions) == 3
        assert len(result.mole_fractions) == 3
        assert abs(sum(result.weight_fractions) - 1.0) < 1e-8

    def test_six_components(self):
        """6成分混合（実務上のデフォルト上限）。"""
        conv = MixtureRatioConverter(
            smiles_list=["CCO", "CCCO", "O", "CC", "CCC", "CCCC"],
            ratio_values=[20, 15, 25, 10, 15, 15],
            input_ratio_type="weight",
        )
        result = conv.convert()
        assert len(result.weight_fractions) == 6
        assert abs(sum(result.mole_fractions) - 1.0) < 1e-8


# ============================================================
# descriptor_weighting_classifier テスト
# ============================================================

class TestDescriptorWeightingClassifier:

    def test_all_rdkit_descriptors_classified(self):
        """RDKit全217記述子が明示的に分類されている。"""
        try:
            from rdkit.Chem import Descriptors
            desc_names = [n for n, _ in Descriptors.descList]
        except ImportError:
            pytest.skip("RDKit not available")

        for name in desc_names:
            assert name in EXPLICIT_DESCRIPTOR_MAP, (
                f"記述子 '{name}' が EXPLICIT_DESCRIPTOR_MAP に未登録"
            )

    def test_all_xtb_base_classified(self):
        """xTB基本記述子10件が全て分類されている。"""
        xtb_names = [
            "xtb_TotalEnergy", "xtb_HomoEnergy", "xtb_LumoEnergy",
            "xtb_HomoLumoGap", "xtb_DipoleMoment", "xtb_Polarizability",
            "xtb_MullikenChargeMax", "xtb_MullikenChargeMin",
            "xtb_MullikenChargeMean", "xtb_MullikenChargeStd",
        ]
        for name in xtb_names:
            assert name in EXPLICIT_DESCRIPTOR_MAP

    def test_all_xtb_ml_classified(self):
        """xTB ML派生特徴量8件が全て分類されている。"""
        ml_names = [
            "xtb_ml_TotalEnergy", "xtb_ml_HomoEnergy", "xtb_ml_LumoEnergy",
            "xtb_ml_Gap", "xtb_ml_Dipole", "xtb_ml_Hardness",
            "xtb_ml_Softness", "xtb_ml_Electrophilicity",
        ]
        for name in ml_names:
            assert name in EXPLICIT_DESCRIPTOR_MAP

    def test_confidence_scores_classified(self):
        """信頼度スコア6件が全てcontextに分類されている。"""
        conf_names = [
            "conf_convergence", "conf_electronic_stability",
            "conf_charge_consistency", "conf_descriptor_completeness",
            "conf_conformer_repr", "conf_overall",
        ]
        for name in conf_names:
            wtype, _ = classify_descriptor(name)
            assert wtype == "context"

    def test_molwt_is_weight(self):
        wtype, _ = classify_descriptor("MolWt")
        assert wtype == "weight"

    def test_homo_is_mole(self):
        wtype, _ = classify_descriptor("xtb_HomoEnergy")
        assert wtype == "mole"

    def test_3d_is_context(self):
        wtype, _ = classify_descriptor("xtb_ml_3D_Asphericity")
        assert wtype == "context"

    def test_unknown_descriptor_fallback(self):
        """未知記述子は正規表現フォールバック → context。"""
        wtype, rationale = classify_descriptor("CompletelyUnknownDescriptor_XYZ")
        assert wtype == "context"
        assert "分類不能" in rationale

    def test_unknown_but_pattern_matched(self):
        """未知だが軌道系パターンにマッチする記述子。"""
        wtype, _ = classify_descriptor("OrbitalEnergy_HOMO_2")
        assert wtype == "mole"

    def test_classify_all(self):
        """一括分類。"""
        names = ["MolWt", "xtb_HomoEnergy", "conf_overall"]
        result = classify_all(names)
        assert result["MolWt"][0] == "weight"
        assert result["xtb_HomoEnergy"][0] == "mole"
        assert result["conf_overall"][0] == "context"

    def test_get_weighting_summary(self):
        names = ["MolWt", "ExactMolWt", "xtb_HomoEnergy", "conf_overall"]
        summary = get_weighting_summary(names)
        assert summary["weight"] == 2
        assert summary["mole"] == 1
        assert summary["context"] == 1

    def test_explicit_map_completeness(self):
        """明示的マッピングの総数が280件以上。"""
        assert len(EXPLICIT_DESCRIPTOR_MAP) >= 280

    def test_all_entries_have_rationale(self):
        """全エントリに根拠文字列がある。"""
        for name, (wtype, rationale) in EXPLICIT_DESCRIPTOR_MAP.items():
            assert wtype in ("weight", "mole", "context"), f"{name}: 不正な分類 '{wtype}'"
            assert len(rationale) > 0, f"{name}: 根拠なし"


# ============================================================
# MixtureFeatureExtractor テスト
# ============================================================

class TestMixtureFeatureExtractor:

    def test_basic_extraction(self):
        """基本的な混合物特徴量抽出。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        ext = MixtureFeatureExtractor()
        result = ext.extract([
            {"smiles": "CCO", "ratio_value": 70.0, "ratio_unit": "weight"},
            {"smiles": "CCCO", "ratio_value": 30.0, "ratio_unit": "weight"},
        ])

        assert len(result.mixture_features) > 0
        assert "mix_MolWt" in result.mixture_features
        assert result.component_features is not None
        assert len(result.component_features) == 2

    def test_molwt_weighted_by_weight(self):
        """MolWtが重量分率で加重平均されている。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        ext = MixtureFeatureExtractor()
        result = ext.extract([
            {"smiles": "CCO", "ratio_value": 50.0, "ratio_unit": "weight"},
            {"smiles": "CCCO", "ratio_value": 50.0, "ratio_unit": "weight"},
        ])

        assert result.weighting_log.get("MolWt") == "weight"
        # MolWtの加重平均は (46.07*0.5 + 60.10*0.5) ≈ 53.08
        assert 40 < result.mixture_features["mix_MolWt"] < 65

    def test_user_override(self):
        """ユーザー上書きが反映される。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        ext = MixtureFeatureExtractor(user_overrides={"MolWt": "mole"})
        result = ext.extract([
            {"smiles": "CCO", "ratio_value": 50.0, "ratio_unit": "weight"},
            {"smiles": "CCCO", "ratio_value": 50.0, "ratio_unit": "weight"},
        ])
        assert result.weighting_log.get("MolWt") == "mole(user)"

    def test_to_dataframe(self):
        """結果を1行DataFrameに変換。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        ext = MixtureFeatureExtractor()
        result = ext.extract([
            {"smiles": "CCO", "ratio_value": 50.0, "ratio_unit": "weight"},
            {"smiles": "O", "ratio_value": 50.0, "ratio_unit": "weight"},
        ])
        df = result.to_dataframe()
        assert df.shape[0] == 1
        assert "mix_MolWt" in df.columns

    def test_conversion_info(self):
        """変換情報が含まれている。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        ext = MixtureFeatureExtractor()
        result = ext.extract([
            {"smiles": "CCO", "ratio_value": 70.0, "ratio_unit": "weight"},
            {"smiles": "O", "ratio_value": 30.0, "ratio_unit": "weight"},
        ])
        info = result.conversion_info
        assert "weight_fractions" in info
        assert "mole_fractions" in info
        assert "molecular_weights" in info
        assert info["input_ratio_type"] == "weight"

    def test_batch_extraction(self):
        """バッチ処理。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        ext = MixtureFeatureExtractor()
        mixtures = [
            [
                {"smiles": "CCO", "ratio_value": 50, "ratio_unit": "weight"},
                {"smiles": "O", "ratio_value": 50, "ratio_unit": "weight"},
            ],
            [
                {"smiles": "c1ccccc1", "ratio_value": 30, "ratio_unit": "weight"},
                {"smiles": "CCO", "ratio_value": 70, "ratio_unit": "weight"},
            ],
        ]
        df = ext.extract_batch(mixtures)
        assert df.shape[0] == 2
        assert "mix_MolWt" in df.columns
