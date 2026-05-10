"""
backend/chem/base.py

化合物特徴量化アダプタの抽象基底クラス。
全アダプタはこのクラスを継承して実装する。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DescriptorResult:
    """
    特徴量化の結果を保持するデータクラス。

    Attributes:
        descriptors: 記述子DataFrame (n_samples x n_descriptors)
        smiles_list: 入力SMILESリスト
        failed_indices: 計算失敗したサンプルのインデックスリスト
        adapter_name: 使用したアダプタ名
        metadata: アダプタ固有の追加情報
    """
    descriptors: pd.DataFrame
    smiles_list: list[str]
    failed_indices: list[int]
    adapter_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """計算成功率を返す。"""
        total = len(self.smiles_list)
        return (total - len(self.failed_indices)) / total if total > 0 else 0.0

    @property
    def n_descriptors(self) -> int:
        """記述子の数を返す。"""
        return self.descriptors.shape[1]


@dataclass
class DescriptorMetadata:
    """
    記述子の詳細属性を保持するデータクラス。
    """
    name: str              # 記述子の内部名
    meaning: str           # 物理的・化学的意味
    is_count: bool         # 整数カウント（原子数、環数等）か
    is_binary: bool = False # バイナリ（有無）か
    description: str = ""  # 詳細説明


class BaseChemAdapter(ABC):
    """
    化合物特徴量化アダプタの抽象基底クラス。

    全ての化合物アダプタはこのクラスを継承し、
    `name`, `description`, `is_available()`, `compute()` を実装すること。

    Implements: 要件定義書 §3.9 化合物特徴量化
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """アダプタの識別名（例: "rdkit", "mordred"）。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """アダプタの説明文（GUI表示用）。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        必要なライブラリがインストールされているかを返す。
        インストールされていない場合は False を返し、例外は送出しない。
        """
        ...

    @abstractmethod
    def compute(
        self,
        smiles_list: list[str],
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILES文字列のリストから記述子を計算して DescriptorResult を返す。

        Args:
            smiles_list: 入力SMILESのリスト
            **kwargs: アダプタ固有のオプション引数

        Returns:
            DescriptorResult インスタンス

        Raises:
            RuntimeError: ライブラリが未インストールで is_available()==False の場合
        """
        ...

    def get_descriptor_names(self) -> list[str]:
        """
        計算可能な記述子名のリストを返す（GUI表示・ルールセット設定用）。
        デフォルト実装はメタデータから名前を抽出して返す。
        """
        metadata = self.get_descriptors_metadata()
        if metadata:
            return [m.name for m in metadata]
        return []

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        """
        各記述子の詳細メタデータ（物理的意味、数え上げ属性等）を返す。
        サブクラスでオーバーライドすること。
        """
        return []

    def _require_available(self) -> None:
        """
        is_available() が False の場合に RuntimeError を送出するヘルパー。
        compute() の冒頭で呼び出すこと。
        """
        if not self.is_available():
            raise RuntimeError(
                f"アダプタ '{self.name}' に必要なライブラリがインストールされていません。"
                f"詳細: {self.description}"
            )

    def __repr__(self) -> str:
        available_str = "✓ available" if self.is_available() else "✗ unavailable"
        return f"{self.__class__.__name__}(name={self.name!r}, {available_str})"
