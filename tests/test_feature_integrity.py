"""
tests/test_feature_integrity.py

必須アダプターがフロントエンドから削除されていないことを検証する保護テスト。
UIリファクタリング等でコードが意図せず消えた場合に即時検出する。
"""
import pathlib
import pytest

# フロントエンド全体を検索対象にする（app.py + pages/）
FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend_streamlit"

# 守るべき必須アダプター（importまたは参照が存在すること）
REQUIRED_ADAPTERS = [
    "RDKitAdapter",
    "MordredAdapter",
    "XTBAdapter",
    "CosmoAdapter",
    "UniPkaAdapter",
    "GroupContribAdapter",
    "MolAIAdapter",
]


def _read_all_frontend_py() -> str:
    """フロントエンド全体の .py ファイルを結合して返す。"""
    texts = []
    for p in FRONTEND_DIR.rglob("*.py"):
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return "\n".join(texts)


class TestFeatureIntegrity:
    """フロントエンドに必須アダプターが含まれることを検証する保護テスト群。"""

    def test_frontend_dir_exists(self):
        assert FRONTEND_DIR.exists(), f"frontend_streamlit が見つかりません: {FRONTEND_DIR}"

    @pytest.mark.parametrize("adapter", REQUIRED_ADAPTERS)
    def test_required_adapter_referenced(self, adapter: str):
        """必須アダプターがフロントエンドのどこかで参照されていること。"""
        content = _read_all_frontend_py()
        assert adapter in content, (
            f"【保護テスト失敗】{adapter} がフロントエンドから消えています。\n"
            f"UIリファクタリング時に誰かが削除した可能性があります。"
        )

    def test_smiles_feature_ui_present(self):
        """SMILES関連のUI設定が存在すること。"""
        content = _read_all_frontend_py()
        # SMILES記述子設定 or SMILES Feature Engineering などの文字列
        has_smiles_ui = any(kw in content for kw in [
            "SMILES記述子設定",
            "SMILES",
            "smiles_transformer",
            "SmilesDescriptorTransformer",
        ])
        assert has_smiles_ui, "SMILES関連のUIブロックがフロントエンドから消えています。"
