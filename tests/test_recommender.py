# -*- coding: utf-8 -*-
"""
tests/test_recommender.py

recommender.py（推奨説明変数データベース）のユニットテスト。

カバー対象:
  - 全目的変数データの整合性（14目的変数×8記述子以上）
  - ユーティリティ関数（名前・カテゴリ検索、一覧取得）
  - DescriptorInfo / TargetRecommendations のフィールド検証
  - エッジケース（空検索、部分一致等）
"""
from __future__ import annotations

import pytest

from backend.chem.recommender import (
    DescriptorInfo,
    TargetRecommendations,
    get_all_target_recommendations,
    get_target_recommendation_by_name,
    get_target_names,
    get_target_categories,
    get_targets_by_category,
    get_all_descriptor_categories,
    get_descriptors_by_category,
)


# ═══════════════════════════════════════════════════════════════════
# データ整合性テスト
# ═══════════════════════════════════════════════════════════════════

class TestRecommenderDatabase:

    def test_all_recommendations_not_empty(self):
        recs = get_all_target_recommendations()
        assert len(recs) > 0

    def test_at_least_14_targets(self):
        """14個の目的変数が登録されていること"""
        recs = get_all_target_recommendations()
        assert len(recs) >= 14

    def test_each_target_has_8_descriptors(self):
        """各目的変数に8つ以上の記述子が定義されていること"""
        for rec in get_all_target_recommendations():
            assert len(rec.descriptors) >= 8, (
                f"'{rec.target_name}' の記述子は {len(rec.descriptors)} 個（8個必要）"
            )

    def test_target_names_unique(self):
        """目的変数名が重複しないこと"""
        names = get_target_names()
        assert len(names) == len(set(names))

    def test_all_descriptors_have_required_fields(self):
        """全記述子が必須フィールドを持つこと"""
        for rec in get_all_target_recommendations():
            for desc in rec.descriptors:
                assert isinstance(desc, DescriptorInfo)
                assert desc.name, f"empty name in {rec.target_name}"
                assert desc.library, f"empty library for {desc.name}"
                assert desc.meaning, f"empty meaning for {desc.name}"
                assert desc.source, f"empty source for {desc.name}"
                assert desc.category, f"empty category for {desc.name}"

    def test_target_recommendations_have_required_fields(self):
        """全TargetRecommendationsが必須フィールドを持つこと"""
        for rec in get_all_target_recommendations():
            assert isinstance(rec, TargetRecommendations)
            assert rec.target_name
            assert rec.summary
            assert rec.category

    def test_known_libraries_used(self):
        """使用ライブラリが既知の範囲内"""
        known = {"RDKit", "XTB", "COSMO-RS", "Uni-pKa", "GroupContribution"}
        for rec in get_all_target_recommendations():
            for desc in rec.descriptors:
                assert desc.library in known, (
                    f"未知ライブラリ '{desc.library}' (記述子={desc.name}, "
                    f"目的変数={rec.target_name})"
                )


# ═══════════════════════════════════════════════════════════════════
# ユーティリティ関数テスト
# ═══════════════════════════════════════════════════════════════════

class TestGetTargetRecommendationByName:

    def test_exact_match(self):
        """完全一致検索"""
        rec = get_target_recommendation_by_name("屈折率")
        assert rec is not None
        assert "屈折率" in rec.target_name

    def test_partial_match(self):
        """部分一致検索"""
        rec = get_target_recommendation_by_name("pKa")
        assert rec is not None

    def test_english_match(self):
        """英語名で検索"""
        rec = get_target_recommendation_by_name("Refractive")
        assert rec is not None

    def test_no_match_returns_none(self):
        """一致なしはNone"""
        rec = get_target_recommendation_by_name("存在しない目的変数XYZ123")
        assert rec is None

    def test_case_insensitive(self):
        """大文字小文字を区別しない"""
        rec = get_target_recommendation_by_name("PKA")
        assert rec is not None


class TestGetTargetNames:

    def test_returns_list_of_strings(self):
        names = get_target_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_count_matches_recommendations(self):
        names = get_target_names()
        recs = get_all_target_recommendations()
        assert len(names) == len(recs)


class TestGetTargetCategories:

    def test_returns_unique_categories(self):
        cats = get_target_categories()
        assert len(cats) == len(set(cats))

    def test_known_categories_present(self):
        cats = get_target_categories()
        expected = {"光・電磁気系", "力学・強度系", "熱・相転移系", "界面・溶液系", "環境・安全性"}
        for exp in expected:
            assert exp in cats, f"カテゴリ '{exp}' が不足"


class TestGetTargetsByCategory:

    def test_returns_correct_category(self):
        targets = get_targets_by_category("光・電磁気系")
        assert len(targets) > 0
        for t in targets:
            assert t.category == "光・電磁気系"

    def test_empty_for_unknown_category(self):
        targets = get_targets_by_category("未知のカテゴリ")
        assert len(targets) == 0


class TestGetAllDescriptorCategories:

    def test_returns_unique_list(self):
        cats = get_all_descriptor_categories()
        assert len(cats) == len(set(cats))

    def test_known_descriptor_categories(self):
        cats = get_all_descriptor_categories()
        expected = {"量子化学・電子状態系", "トポロジー系", "極性・官能基系",
                    "熱力学・相互作用系", "立体・形状系"}
        for exp in expected:
            assert exp in cats, f"記述子カテゴリ '{exp}' が不足"


class TestGetDescriptorsByCategory:

    def test_returns_descriptors(self):
        descs = get_descriptors_by_category("量子化学・電子状態系")
        assert len(descs) > 0
        for d in descs:
            assert d.category == "量子化学・電子状態系"

    def test_no_duplicates(self):
        """同じカテゴリ内で記述子名が重複しないこと"""
        descs = get_descriptors_by_category("極性・官能基系")
        names = [d.name for d in descs]
        assert len(names) == len(set(names))

    def test_empty_for_unknown(self):
        descs = get_descriptors_by_category("未知カテゴリ")
        assert len(descs) == 0


# ═══════════════════════════════════════════════════════════════════
# 特定目的変数の記述子検証
# ═══════════════════════════════════════════════════════════════════

class TestSpecificTargets:

    def test_refractive_index_has_molmr(self):
        """屈折率にMolMRが含まれる"""
        rec = get_target_recommendation_by_name("屈折率")
        names = [d.name for d in rec.descriptors]
        assert "MolMR" in names

    def test_tg_has_backbone_flexibility(self):
        """TgにBackboneFlexibilityが含まれる"""
        rec = get_target_recommendation_by_name("Tg")
        names = [d.name for d in rec.descriptors]
        assert "BackboneFlexibility" in names

    def test_toxicity_has_logp(self):
        """毒性にLogPが含まれる"""
        rec = get_target_recommendation_by_name("毒性")
        names = [d.name for d in rec.descriptors]
        assert "MolLogP" in names

    def test_viscosity_has_molwt(self):
        """粘度にMolWtが含まれる"""
        rec = get_target_recommendation_by_name("粘度")
        names = [d.name for d in rec.descriptors]
        assert "MolWt" in names
