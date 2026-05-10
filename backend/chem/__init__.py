"""backend/chem/__init__.py"""
from backend.chem.base import BaseChemAdapter, DescriptorResult


def _make_unavailable_adapter(name: str):
    """ライブラリ未インストール時のダミーアダプタークラスを動的生成する。"""
    class _UnavailableAdapter:
        _adapter_name = name
        def __init__(self, **kwargs):
            pass
        def is_available(self) -> bool:
            return False
        def compute(self, *args, **kwargs):
            raise RuntimeError(
                f"{self._adapter_name} は利用できません。"
                "必要なライブラリをインストールしてください。"
            )
    _UnavailableAdapter.__name__ = name
    _UnavailableAdapter.__qualname__ = name
    return _UnavailableAdapter


# ── 各アダプターを安全import ──────────────────────────────────

try:
    from backend.chem.rdkit_adapter import RDKitAdapter
except Exception:
    RDKitAdapter = _make_unavailable_adapter("RDKitAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.xtb_adapter import XTBAdapter
except Exception:
    XTBAdapter = _make_unavailable_adapter("XTBAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.cosmo_adapter import CosmoAdapter
except Exception:
    CosmoAdapter = _make_unavailable_adapter("CosmoAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.unipka_adapter import UniPkaAdapter
except Exception:
    UniPkaAdapter = _make_unavailable_adapter("UniPkaAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.group_contrib_adapter import GroupContribAdapter
except Exception:
    GroupContribAdapter = _make_unavailable_adapter("GroupContribAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.mordred_adapter import MordredAdapter
except Exception:
    MordredAdapter = _make_unavailable_adapter("MordredAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.molai_adapter import MolAIAdapter
except Exception:
    MolAIAdapter = _make_unavailable_adapter("MolAIAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.uma_adapter import UMAAdapter
except Exception:
    UMAAdapter = _make_unavailable_adapter("UMAAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.skfp_adapter import SkfpAdapter
except Exception:
    SkfpAdapter = _make_unavailable_adapter("SkfpAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.padel_adapter import PaDELAdapter
except Exception:
    PaDELAdapter = _make_unavailable_adapter("PaDELAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
except Exception:
    DescriptaStorusAdapter = _make_unavailable_adapter("DescriptaStorusAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.mol2vec_adapter import Mol2VecAdapter
except Exception:
    Mol2VecAdapter = _make_unavailable_adapter("Mol2VecAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.molfeat_adapter import MolfeatAdapter
except Exception:
    MolfeatAdapter = _make_unavailable_adapter("MolfeatAdapter")  # type: ignore[assignment,misc]

try:
    from backend.chem.chemprop_adapter import ChempropAdapter
except Exception:
    ChempropAdapter = _make_unavailable_adapter("ChempropAdapter")  # type: ignore[assignment,misc]

__all__ = [
    "BaseChemAdapter",
    "DescriptorResult",
    "RDKitAdapter",
    "XTBAdapter",
    "CosmoAdapter",
    "UniPkaAdapter",
    "GroupContribAdapter",
    "MordredAdapter",
    "MolAIAdapter",
    "UMAAdapter",
    "SkfpAdapter",
    "PaDELAdapter",
    "DescriptaStorusAdapter",
    "Mol2VecAdapter",
    "MolfeatAdapter",
    "ChempropAdapter",
    "get_available_adapters",
    "ADAPTER_REGISTRY",
]


# ── アダプターレジストリ ─────────────────────────
ADAPTER_REGISTRY: dict[str, type] = {
    "RDKit": RDKitAdapter,
    "Mordred": MordredAdapter,
    "scikit-FP": SkfpAdapter,
    "Mol2Vec": Mol2VecAdapter,
    "GroupContrib": GroupContribAdapter,
    "MolAI": MolAIAdapter,
    "UMA": UMAAdapter,
    "PaDEL": PaDELAdapter,
    "DescriptaStorus": DescriptaStorusAdapter,
    "Molfeat": MolfeatAdapter,
    "Chemprop": ChempropAdapter,
    "XTB": XTBAdapter,
    "COSMO": CosmoAdapter,
    "UniPKa": UniPkaAdapter,
}


def get_available_adapters() -> dict[str, type]:
    """インストール済みで利用可能なアダプターのみを返す。

    Returns
    -------
    dict[str, type]
        {エンジン名: アダプタークラス} の辞書。
        is_available() が True を返すもののみ含む。
    """
    result = {}
    for name, cls in ADAPTER_REGISTRY.items():
        try:
            instance = cls()
            if hasattr(instance, "is_available") and instance.is_available():
                result[name] = cls
            elif not hasattr(instance, "is_available"):
                # is_available メソッドがない → 利用可能とみなす
                result[name] = cls
        except Exception:
            pass
    return result

