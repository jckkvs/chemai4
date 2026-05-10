"""
tests/test_recommender_comprehensive.py

recommender.py の包括テスト。
DescriptorInfo / TargetRecommendations / 全ヘルパー関数を網羅。
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


class TestDescriptorInfo:
    def test_fields(self):
        d = DescriptorInfo(
            name="MolWt", library="RDKit",
            meaning="分子量", source="General", category="立体・形状系"
        )
        assert d.name == "MolWt"
        assert d.library == "RDKit"


class TestTargetRecommendations:
    def test_fields(self):
        t = TargetRecommendations(
            target_name="テスト",
            summary="テスト用",
            category="テストカテゴリ",
            descriptors=[],
        )
        assert t.target_name == "テスト"


class TestGetAllRecommendations:
    def test_returns_list(self):
        recs = get_all_target_recommendations()
        assert isinstance(recs, list)
        assert len(recs) > 10  # 多数の目的変数が登録済み

    def test_each_has_descriptors(self):
        recs = get_all_target_recommendations()
        for rec in recs:
            assert len(rec.descriptors) >= 8  # 「8個以上」のルール


class TestGetByName:
    def test_exact(self):
        rec = get_target_recommendation_by_name("屈折率")
        assert rec is not None
        assert "屈折率" in rec.target_name

    def test_partial(self):
        rec = get_target_recommendation_by_name("Tg")
        assert rec is not None

    def test_not_found(self):
        rec = get_target_recommendation_by_name("存在しない目的変数xyz")
        assert rec is None


class TestGetTargetNames:
    def test_returns_list(self):
        names = get_target_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


class TestGetTargetCategories:
    def test_unique(self):
        cats = get_target_categories()
        assert len(cats) == len(set(cats))

    def test_known_categories(self):
        cats = get_target_categories()
        assert "光・電磁気系" in cats
        assert "力学・強度系" in cats


class TestGetTargetsByCategory:
    def test_filter(self):
        recs = get_targets_by_category("光・電磁気系")
        assert len(recs) > 0
        for rec in recs:
            assert rec.category == "光・電磁気系"

    def test_empty(self):
        recs = get_targets_by_category("存在しないカテゴリ")
        assert recs == []


class TestGetDescriptorCategories:
    def test_returns(self):
        cats = get_all_descriptor_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_known(self):
        cats = get_all_descriptor_categories()
        assert "量子化学・電子状態系" in cats


class TestGetDescriptorsByCategory:
    def test_filter(self):
        descs = get_descriptors_by_category("量子化学・電子状態系")
        assert len(descs) > 0
        for d in descs:
            assert d.category == "量子化学・電子状態系"

    def test_unique_names(self):
        descs = get_descriptors_by_category("量子化学・電子状態系")
        names = [d.name for d in descs]
        assert len(names) == len(set(names))  # 重複なし
