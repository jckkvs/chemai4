"""
backend/chem/descriptor_sets.py

記述子セット管理。
セット = 名前 + エンジンフラグ辞書。
セッションへの保持とJSONファイルへの永続化を提供する。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# バックエンドが認識するエンジンフラグキー（smiles_transformer.py L325-337 + L361と一致）
ENGINE_FLAG_KEYS: list[str] = [
    "use_mordred", "use_xtb", "use_cosmo", "use_unipka",
    "use_contrib", "use_uma", "use_skfp", "use_padel",
    "use_ds", "use_mol2vec", "use_molfeat", "use_chemprop",
    "use_molai",
]

# エンジンフラグキー → 人間可読ラベル
ENGINE_LABELS: dict[str, str] = {
    "use_rdkit":    "RDKit",
    "use_mordred":  "Mordred",
    "use_xtb":      "xTB",
    "use_cosmo":    "COSMO-RS",
    "use_unipka":   "UniPKa",
    "use_contrib":  "GroupContrib",
    "use_uma":      "UMA",
    "use_skfp":     "scikit-FP",
    "use_padel":    "PaDEL",
    "use_ds":       "DescriptaStorus",
    "use_mol2vec":  "Mol2Vec",
    "use_molfeat":  "Molfeat",
    "use_chemprop": "Chemprop",
    "use_molai":    "MolAI",
}


@dataclass
class DescriptorSet:
    """記述子の組み合わせ（セット）1つ分を表す。"""
    name: str
    engine_flags: dict[str, bool] = field(default_factory=dict)
    molai_n_components: int = 32
    enabled: bool = True  # 一括評価に含めるか

    @property
    def active_engines(self) -> list[str]:
        """ONになっているエンジンのラベルリスト。RDKitは常にON。"""
        labels = ["RDKit"]
        for k, v in self.engine_flags.items():
            if v and k in ENGINE_LABELS:
                labels.append(ENGINE_LABELS[k])
        return labels

    @property
    def summary(self) -> str:
        """'RDKit + MolAI + XTB' のような短い要約文字列。"""
        return " + ".join(self.active_engines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DescriptorSet":
        return cls(
            name=d["name"],
            engine_flags=d.get("engine_flags", {}),
            molai_n_components=d.get("molai_n_components", 32),
            enabled=d.get("enabled", True),
        )


# ── デフォルトプリセット ────────────────────────────────────
DEFAULT_SETS: list[DescriptorSet] = [
    DescriptorSet(
        name="🧪 基本（RDKitのみ）",
        engine_flags={},
    ),
    DescriptorSet(
        name="💊 活性予測（RDKit + FP）",
        engine_flags={"use_skfp": True},
    ),
    DescriptorSet(
        name="🔬 網羅的QSPR",
        engine_flags={"use_mordred": True},
    ),
    DescriptorSet(
        name="🧠 深層学習（MolAI）",
        engine_flags={"use_molai": True},
    ),
    DescriptorSet(
        name="⚛️ 量子化学込み",
        engine_flags={"use_xtb": True, "use_cosmo": True},
    ),
    DescriptorSet(
        name="⚛️+ML 量子化学+派生特徴量",
        engine_flags={"use_xtb": True},
    ),
]


class DescriptorSetManager:
    """記述子セットのCRUD管理。"""

    _SAVE_DIR = Path.home() / ".chemai" / "descriptor_sets"

    def __init__(self, sets: list[DescriptorSet] | None = None):
        self._sets: list[DescriptorSet] = sets if sets is not None else []

    # ── CRUD ────────────────────────────────────────────────
    def add(self, ds: DescriptorSet) -> None:
        """セットを追加。同名は上書き。"""
        self._sets = [s for s in self._sets if s.name != ds.name]
        self._sets.append(ds)

    def remove(self, name: str) -> None:
        self._sets = [s for s in self._sets if s.name != name]

    def get(self, name: str) -> DescriptorSet | None:
        for s in self._sets:
            if s.name == name:
                return s
        return None

    def list_all(self) -> list[DescriptorSet]:
        return list(self._sets)

    def list_enabled(self) -> list[DescriptorSet]:
        return [s for s in self._sets if s.enabled]

    def duplicate(self, name: str, new_name: str) -> DescriptorSet | None:
        src = self.get(name)
        if src is None:
            return None
        dup = DescriptorSet(
            name=new_name,
            engine_flags=dict(src.engine_flags),
            molai_n_components=src.molai_n_components,
            enabled=True,
        )
        self.add(dup)
        return dup

    def reorder(self, names: list[str]) -> None:
        by_name = {s.name: s for s in self._sets}
        self._sets = [by_name[n] for n in names if n in by_name]
        for s in self._sets:
            if s.name not in names:
                self._sets.append(s)

    # ── 永続化 ──────────────────────────────────────────────
    def save_to_file(self, filename: str = "sets.json") -> Path:
        self._SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = self._SAVE_DIR / filename
        data = [s.to_dict() for s in self._sets]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_from_file(cls, filename: str = "sets.json") -> "DescriptorSetManager":
        path = cls._SAVE_DIR / filename
        if not path.exists():
            return cls(sets=list(DEFAULT_SETS))
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sets = [DescriptorSet.from_dict(d) for d in data]
            return cls(sets=sets)
        except Exception:
            return cls(sets=list(DEFAULT_SETS))

    # ── セッション保存/復元 ─────────────────────────────────
    def to_session(self) -> list[dict]:
        return [s.to_dict() for s in self._sets]

    @classmethod
    def from_session(cls, data: list[dict]) -> "DescriptorSetManager":
        sets = [DescriptorSet.from_dict(d) for d in data]
        return cls(sets=sets)
